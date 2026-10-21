#!/usr/bin/env python3
"""Explicit, bounded recovery of a memory-off final pair after a first-call failure.

Keeps the existing ledger, response fingerprints, budgets, memory databases and
unknown request. It deliberately does not change the normal conservative resume
policy. A completed failure and its terminal control snapshot are archived before
reopening; successful tasks are never reopened. No model request without --execute.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from LangChain.medqa import cli
from LangChain.medqa.checkpoint import Checkpoint, CheckpointError, DurableCalls, DurableTools, immutable_manifest
from LangChain.medqa.data import Dataset
from LangChain.medqa.runtime import Runtime


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan_recovery(store, condition, questions):
    conditions = [m + '_' + k for m, k in store.meta('conditions')]
    if condition not in conditions or not condition.endswith('_off'):
        raise CheckpointError('此恢复器仅支持原批次的无记忆条件；不会重建或修改记忆库')
    ids = [q.id for q in questions]
    for other in conditions:
        if other != condition and [r['question_id'] for r in store.results(other)] != ids:
            raise CheckpointError('其他条件尚未完整结束；不适用此有界恢复器')
    rows = store.db.execute('SELECT * FROM questions WHERE condition=? ORDER BY position', (condition,)).fetchall()
    if len(ids) < 2 or len(rows) != len(ids) - 1:
        raise CheckpointError('只允许恢复倒数第二题失败、最后一题未执行的批次')
    for index, row in enumerate(rows):
        if row['position'] != index or row['question_id'] != ids[index] or row['state'] != 'complete':
            raise CheckpointError('题序、状态或已完成前缀异常')
        result = json.loads(row['result'])
        if index < len(rows) - 1 and result.get('answer') is None:
            raise CheckpointError('前缀仍有未回答题；禁止静默跳过')
    failed = rows[-1]
    result = json.loads(failed['result'])
    if result.get('answer') is not None or result.get('status') != 'runtime_error':
        raise CheckpointError('只恢复无答案的运行错误，不能重做已有答案')
    operations = store.db.execute('SELECT * FROM operations WHERE condition=? AND position=?',
                                 (condition, failed['position'])).fetchall()
    empty_views_only = all(r['kind'] == 'evidence_view' and r['state'] == 'complete'
        and not json.loads(r['request']).get('documents')
        and not json.loads(r['result']).get('views') for r in operations)
    if json.loads(failed['writes']) or not empty_views_only:
        raise CheckpointError('失败题已有工具操作或记忆写入，不适用首请求恢复')
    calls = store.db.execute('SELECT * FROM calls WHERE condition=? AND position=?',
                             (condition, failed['position'])).fetchall()
    if (len(calls) != 1 or calls[0]['state'] != 'pending' or calls[0]['ordinal'] != 1
            or calls[0]['role'] != 'Planner' or calls[0]['raw'] is not None
            or store.db.execute("SELECT count(*) FROM calls WHERE state IN ('pending','abandoned')").fetchone()[0] != 1):
        raise CheckpointError('仅允许一个未保存响应的首个 Planner 请求，其他未知请求保持阻断')
    controls = store.db.execute('SELECT * FROM control_states WHERE condition=? AND position=?',
                                (condition, failed['position'])).fetchall()
    if any(r['trigger'] != 'question_closed' for r in controls):
        raise CheckpointError('发现非终止状态快照，不能重置决策历史')
    return {'condition': condition, 'positions': [len(ids)-2, len(ids)-1],
            'question_ids': ids[-2:], 'failed_row': dict(failed),
            'terminal_control_states': [dict(r) for r in controls],
            'unknown_request': {k: calls[0][k] for k in ('condition','position','ordinal','attempt','role','state')},
            'previous_failed_result': result}


def reopen_failure(store, plan):
    history = store.meta('bounded_recovery_history', [])
    history.append({'authorized_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), **plan})
    with store.db:
        store.set_meta('bounded_recovery_history', history)
        store.db.execute("UPDATE questions SET state='running',result=NULL,writes=NULL WHERE condition=? AND position=?",
                         (plan['condition'], plan['positions'][0]))
        # Archive above retains the failed terminal snapshot verbatim. It is not
        # an executable successful decision prefix and must not replace the retry.
        store.db.execute('DELETE FROM control_states WHERE condition=? AND position=?',
                         (plan['condition'], plan['positions'][0]))


def publish(store, out, questions):
    report, ids = {}, {}
    for mode, memory in store.meta('conditions'):
        name = mode + '_' + memory
        rows = store.results(name)
        store.export_results(name)
        report[name] = cli.summarize(rows)
        report[name]['resume_count'] = store.meta('resume_count', 0)
        ids[name] = [r['question_id'] for r in rows]
    cli.dump(out/'report.json', report)
    cli.dump(out/'comparison.json', cli.compare_reports(report, ids, [q.id for q in questions]))
    cli.dump(out/'recovery.json', {'resume_count': store.meta('resume_count', 0),
        'unknown_calls': sum(store.unknown_calls(c) for c in report),
        'latency_comparable': False, 'policy': 'explicit bounded final-pair recovery',
        'note': '原未知请求保留，不当作零成本；仅继续已授权的最后两题，不扩大为自动重试。'})


def merge_failed_attempt(result, previous):
    # First-call failure had no response or tools. Preserve its known request
    # counters and duration, without inventing prompt/completion token usage.
    old = previous['metrics']
    for key in ('model_calls', 'messages', 'communication_chars', 'request_chars',
                'request_bytes', 'system_chars', 'system_bytes',
                'protocol_control_chars', 'protocol_control_bytes', 'protocol_maps',
                'usage_missing_calls', 'elapsed_seconds'):
        result['metrics'][key] = result['metrics'].get(key, 0) + old.get(key, 0)
    result['metrics']['recovery_previous_failed_seconds'] = old['elapsed_seconds']
    result['events'] = deepcopy(previous['events']) + [
        {'tool': 'authorized_failed_task_recovery', 'original_failure_retained': True,
         'unknown_cost_retained': True}] + result['events']


def run(args, data, textbooks, agents, store, out, questions, plan):
    condition = plan['condition']
    mode = condition.rsplit('_', 1)[0]
    for position in plan['positions']:
        total = sum(r['metrics']['total_tokens'] for m,k in store.meta('conditions')
                    for r in store.results(m+'_'+k))
        used = sum(r['metrics']['total_tokens'] for r in store.results(condition))
        if total >= args.max_total_tokens or used >= args.max_condition_tokens:
            raise CheckpointError('原累计 Token 保护预算已达上限')
        q = questions[position]
        row = store.begin(condition, position, q.id)
        calls = DurableCalls(agents, store, condition, position, retry_unknown=True, max_calls=args.max_calls)
        tools = DurableTools(store, condition, position, max_searches=args.max_searches, max_reads=args.max_reads)
        previous = plan['previous_failed_result'] if position == plan['positions'][0] else None
        if previous:
            calls.historical_seconds = max(calls.historical_seconds, previous['metrics']['elapsed_seconds'])
        runtime = Runtime(calls, textbooks, memory=None, embedder=None, mode=mode,
            max_calls=args.max_calls, memory_threshold=args.memory_threshold,
            max_prompt_chars=args.max_prompt_chars, context_profile=args.context_profile,
            max_question_tokens=args.max_question_tokens, deadline_seconds=args.deadline_seconds,
            task_group=args.task_group, memory_write=args.memory_write == 'on',
            max_searches=args.max_searches, max_reads=args.max_reads,
            max_stalled_rounds=args.max_stalled_rounds, max_contract_repairs=args.max_contract_repairs,
            evidence_processor=args.evidence_processor == 'on', evidence_span_chars=args.evidence_span_chars,
            journal=tools, output_policy=args.output_policy, memory_policy=args.memory_policy,
            memory_selection=args.memory_selection, memory_recall_limit=args.memory_recall_limit)
        result = runtime.run(q, trace_id=row['trace'])
        if calls.fatal_error or tools.fatal_error:
            raise calls.fatal_error or tools.fatal_error
        calls.validate_replay()
        result['correct'] = data.grade(result)
        if previous:
            merge_failed_attempt(result, previous)
        result['metrics'].update(checkpoint_replayed_calls=calls.replayed,
            checkpoint_replayed_operations=tools.replayed, checkpoint_executed_operations=tools.executed,
            checkpoint_unknown_recomputations=tools.unknown_recomputations,
            checkpoint_search_attempts=tools.attempts('search_textbooks'),
            checkpoint_source_read_attempts=tools.attempts('read_source'),
            checkpoint_unknown_calls=store.unknown_calls(condition, position),
            checkpoint_resumed=previous is not None, checkpoint_request_attempts=calls.request_attempts())
        store.save_result(condition, position, result, [])
        store.apply_memory(condition, None)
        publish(store, out, questions)
        print(condition, position+1, q.id, result['status'], 'correct='+str(result['correct']), flush=True)
        if result.get('answer') is None:
            raise CheckpointError('补跑未产生答案；已保存所有记录，停止，不自动循环请求')
        if store.db.execute("SELECT count(*) FROM calls WHERE state='pending'").fetchone()[0]:
            raise CheckpointError('补跑产生新的未知请求，立即停止')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--condition', default='text_off')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--retry-unknown-calls', action='store_true')
    options = parser.parse_args()
    if options.execute and not options.retry_unknown_calls:
        parser.error('执行必须显式接受未知请求可能重复计费：--retry-unknown-calls')
    out = options.output.resolve()
    _, args = cli.parse_args(['compare','--resume','--output',str(out),'--retry-unknown-calls'])
    args.output = out
    data = Dataset(args.benchmark)
    questions = cli.select_questions(args, data)
    textbooks = cli.books(args)
    try:
        agents, model = cli.create_model(args)  # Constructs client; no generation yet.
        with Checkpoint(out, resume=True) as store:
            manifest = deepcopy(store.meta('manifest'))
            manifest.update(settings={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()},
                model=model, source_sha256=cli.source_hashes(), dataset_sha256=data.sha256,
                textbooks_version=textbooks.version,
                retrieval_version=getattr(textbooks,'operation_version',type(textbooks).__name__),
                question_ids=[q.id for q in questions])
            if immutable_manifest(manifest) != immutable_manifest(store.meta('manifest')):
                raise CheckpointError('源码、配置、模型、数据或检索版本变化，拒绝恢复')
            plan = plan_recovery(store, options.condition, questions)
            print(json.dumps({k:v for k,v in plan.items() if k not in
                  {'failed_row','terminal_control_states','previous_failed_result'}},ensure_ascii=False),flush=True)
            if not options.execute:
                return
            backup = out/('before-bounded-recovery-'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S'))
            backup.mkdir(exist_ok=False)
            for path in out.iterdir():
                if path.is_file() and path.suffix in {'.json','.jsonl','.sqlite3','.md'}:
                    if path.suffix == '.sqlite3':
                        with sqlite3.connect(backup/path.name) as destination:
                            if path.name == 'checkpoint.sqlite3':
                                store.db.backup(destination)
                            else:
                                with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as source:
                                    source.backup(destination)
                    else:
                        shutil.copy2(path,backup/path.name)
            preserved = {str(p):digest(p) for p in out.glob('*_on.sqlite3')}
            unaffected = {tuple(row[:2]): row[2] for row in store.db.execute(
                'SELECT condition,position,result FROM questions WHERE NOT(condition=? AND position=?)',
                (plan['condition'],plan['positions'][0]))}
            store.initialize(manifest, store.meta('conditions'), resume=True)
            reopen_failure(store,plan)
            audit = {'backup':str(backup),'question_ids':plan['question_ids'],
                     'preserved_memory_sha256':preserved,'source_sha256':cli.source_hashes(),
                     'authorized_unknown_requests':1,'max_new_tasks':2,'state':'running'}
            cli.dump(out/'bounded-recovery.json',audit)
            publish(store,out,questions)
            try:
                textbooks.warmup()
                run(args,data,textbooks,agents,store,out,questions,plan)
                assert all(digest(Path(p)) == value for p,value in preserved.items()), 'Memory database changed'
                assert all(store.question(c,p)['result'] == value for (c,p),value in unaffected.items()), 'Existing result changed'
                audit.update(state='completed',memory_databases_unchanged=True,successful_results_unchanged=True,
                    unknown_request_cost_still_unknown=True)
            except BaseException as error:
                audit.update(state='stopped',error_type=type(error).__name__)
                raise
            finally:
                cli.dump(out/'bounded-recovery.json',audit)
                publish(store,out,questions)
    finally:
        textbooks.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('恢复停止：',str(error) if isinstance(error,CheckpointError) else type(error).__name__,file=sys.stderr)
        sys.exit(1)
