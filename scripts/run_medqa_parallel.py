#!/usr/bin/env python3
"""四条件独立进程并行；每个条件内部严格按原题序运行。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from LangChain.medqa import cli
from LangChain.medqa.data import Dataset
from summarize_medqa_compare import summarize

CONDITIONS = ('text_off', 'structured_off', 'text_on', 'structured_on')


def check_local_service(model, base_url, api_key):
    """只查询模型列表，不生成Token；失败时不创建实验目录。"""
    from urllib.request import Request, build_opener, ProxyHandler
    from urllib.error import URLError
    request = Request(base_url.rstrip('/') + '/models',
                      headers={'Authorization': 'Bearer ' + api_key})
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=10) as response:
            available = json.load(response)
        names = {entry['id'] for entry in available['data']}
    except (URLError, OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit('本地模型服务不可用（' + type(error).__name__ +
                         '）；先在另一个终端启动 bash scripts/serve_qwen14b_gpu0.sh，'
                         '等待 Application startup complete 后再运行评测。本次未启动子任务。') from None
    if model not in names:
        raise SystemExit('服务未提供配置模型 ' + model + '；请核对端口与模型名。本次未启动子任务。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--all', action='store_true')
    choice.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, choices=(1, 2, 4), default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=Path, help='覆盖配置中的输出目录；失败记录保留，重测使用新目录')
    args = parser.parse_args()
    os.chdir(ROOT)
    scope = ['--all'] if args.all else ['--limit', str(args.limit)]
    overrides = ['--output', str(args.output)] if args.output else []
    _, settings = cli.parse_args(['compare', '--config', str(args.config), *scope, *overrides])
    if args.limit is not None and args.limit < 1:
        parser.error('limit必须大于0')
    if settings.memory_source:
        parser.error('第一版并行入口仅支持当前在线积累实验；冻结库仍使用原入口')
    model = cli.model_connection(settings)
    from urllib.parse import urlparse
    if urlparse(model[1]).hostname not in {'127.0.0.1', 'localhost', '::1'}:
        parser.error('并行入口只允许本地模型，不自动发起并发云API实验')
    if not args.dry_run:
        check_local_service(*model[:3])
    out = settings.output
    if out is None:
        parser.error('配置须明确独立输出目录')
    if args.resume:
        if not (out / 'parallel-plan.json').is_file():
            parser.error('未找到并行实验计划，不能续跑')
    else:
        out.mkdir(parents=True, exist_ok=False)
    with (out / '.parallel.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = Dataset(settings.benchmark)
        questions = cli.select_questions(settings, data)
        ids = [q.id for q in questions]
        if not ids:
            parser.error('没有选中题目')
        source = cli.source_hashes()
        import hashlib
        runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        plan = {'config': json.loads(args.config.read_text()), 'workers': args.workers,
                'output': str(out.resolve()),
                'question_ids': ids, 'source_sha256': source, 'runner_sha256': runner_sha,
                'scope': scope, 'same_gpu_latency_comparable': False}
        plan_path = out / 'parallel-plan.json'
        if args.resume:
            if json.loads(plan_path.read_text()) != plan:
                parser.error('配置、题序、源码或并发数已变化，不能续跑')
        else:
            cli.dump(plan_path, plan)
            files = [*source, str(Path(__file__).relative_to(ROOT)), 'scripts/run_medqa.sh',
                     'scripts/serve_qwen14b_gpu0.sh', 'scripts/summarize_medqa_compare.py']
            for filename in files:
                target = out / 'source' / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / filename).read_bytes())
        for folder in ('configs', 'logs', 'conditions'):
            (out / folder).mkdir(exist_ok=True)
        configs = {}
        for condition in CONDITIONS:
            mode, memory = condition.split('_')
            cfg = dict(plan['config'], mode=mode, memory=memory,
                       output=str(out / 'conditions' / condition))
            configs[condition] = out / 'configs' / (condition + '.json')
            cli.dump(configs[condition], cfg)

        def publish(active, exits):
            reports, completed = {}, {}
            for condition in CONDITIONS:
                folder = out / 'conditions' / condition
                if not (folder / 'report.json').is_file():
                    continue
                item = json.loads((folder / 'report.json').read_text()).get(condition)
                if item is None:
                    continue
                reports[condition] = item
                path = folder / (condition + '.jsonl')
                text = path.read_text() if path.is_file() else ''
                rows = [json.loads(line) for line in text.splitlines() if line.strip()]
                done = [r['question_id'] for r in rows]
                if done != ids[:len(done)]:
                    raise RuntimeError('子实验题序不匹配')
                completed[condition] = done
                cli.atomic_text(out / (condition + '.jsonl'), text)
                if not (out / 'manifest.json').is_file():
                    manifest = json.loads((folder / 'manifest.json').read_text())
                    manifest['parallel_execution'] = {'workers': args.workers, 'shared_gpu': 0,
                        'source_directories': {c: str(out / 'conditions' / c) for c in CONDITIONS},
                        'note': '各条件独立检查点和记忆库；跨条件共享GPU，单题时延不作独立条件收益比较。'}
                    cli.dump(out / 'manifest.json', manifest)
            cli.dump(out / 'report.json', reports)
            comparison = cli.compare_reports(reports, completed, ids)
            comparison['latency_comparison_valid'] = False
            comparison['latency_note'] = '并发共享GPU且包含排队；仅报告观测时延和整批吞吐，不据此推断协议或记忆的独立时延收益。'
            for value in comparison.values():
                if isinstance(value, dict) and 'latency_reduction' in value:
                    value['latency_reduction'] = None
            cli.dump(out / 'comparison.json', comparison)
            status = {'target_per_condition': len(ids),
                      'completed': {c: len(completed.get(c, [])) for c in CONDITIONS},
                      'active_pids': {c: p.pid for c, p in active.items()}, 'exit_codes': exits,
                      'matched_complete': comparison['matched_complete'],
                      'state': ('completed' if comparison['matched_complete'] and not active else
                                'failed' if exits and not active and not comparison['matched_complete'] else
                                'running' if active else 'pending'),
                      'updated_unix': time.time()}
            cli.dump(out / 'run-status.json', status)
            return completed

        if args.dry_run:
            print(json.dumps({'output': str(out), 'conditions': list(CONDITIONS),
                              'questions_per_condition': len(ids), 'workers': args.workers,
                              'model_requests': 0}, ensure_ascii=False), flush=True)
            return
        queue, active, exits = list(CONDITIONS), {}, {}
        done = publish(active, exits)
        queue = [c for c in queue if done.get(c) != ids]
        started = time.monotonic()
        while queue or active:
            while queue and len(active) < args.workers:
                condition = queue.pop(0)
                child = out / 'conditions' / condition
                command = ['bash', 'scripts/run_medqa.sh', 'run', '--config', str(configs[condition]), *scope]
                if child.exists():
                    if not args.resume:
                        raise RuntimeError('子目录已存在，拒绝重复执行')
                    command += ['--resume', '--output', str(child)]
                with (out / 'logs' / (condition + '.log')).open('ab') as stream:
                    active[condition] = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONUNBUFFERED': '1'},
                        start_new_session=True)
                print('started', condition, active[condition].pid, flush=True)
            time.sleep(5)
            for condition, process in list(active.items()):
                code = process.poll()
                if code is not None:
                    exits[condition] = code
                    del active[condition]
                    if code:
                        queue.clear()  # 不再启动排队任务；已运行子实验保留其检查点。
            done = publish(active, exits)
        cli.dump(out / 'execution.json', {'wall_seconds_this_session': time.monotonic() - started,
            'resume': args.resume, 'workers': args.workers, 'exit_codes': exits,
            'note': '计时含启动与预热；LLM精确Token以各条件usage为准，额外测速单独记录。'})
        if (out / 'manifest.json').is_file():
            cli.atomic_text(out / 'summary.md', summarize(out))
        if any(code != 0 for code in exits.values()) or any(done.get(c) != ids for c in CONDITIONS):
            print('实验未完成；已保存部分结果：', out, file=sys.stderr, flush=True)
            print('各组完成数：', {c: len(done.get(c, [])) for c in CONDITIONS}, file=sys.stderr, flush=True)
            print('请查看 logs/*.log；进程退出码为0也不代表题目全部完成。', file=sys.stderr, flush=True)
            raise SystemExit(1)
        print('四组全量已完成。results:', out, flush=True)


if __name__ == '__main__':
    main()
