#!/usr/bin/env python3
"""Read-only experiment monitor; refresh summaries without calling any model."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from LangChain.medqa.cli import summarize
from summarize_medqa_compare import summarize as detailed_summary, alive, CONDITIONS, LABELS


def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.monitor.tmp');temp.write_text(value,encoding='utf-8');os.replace(temp,path)


def refresh(run):
    manifest=json.loads((run/'manifest.json').read_text())
    slices=json.loads((ROOT/'experiments/datasets/medical-related-50-v2/slices.json').read_text())
    rows={c:[json.loads(s) for s in (run/(c+'.jsonl')).read_text().splitlines() if s.strip()]
          if (run/(c+'.jsonl')).exists() else [] for c in CONDITIONS}
    ids=manifest['question_ids'];complete=all([r['question_id'] for r in rows[c]]==ids for c in CONDITIONS)
    summaries={name:{c:summarize([r for r in rows[c] if r['question_id'] in group]) for c in CONDITIONS}
               for name,group in {'all50':ids,**slices}.items()}
    lines=['# 50题四条件实验进度与分层结果','',
           '刷新时间：'+datetime.now(timezone.utc).isoformat(),'',
           f"状态：{'200/200完成' if complete else str(sum(map(len,rows.values())))+'/200已完成'}。下表仅为已完成任务；跨条件收益仅在相同题目全部完成时比较。",'',
           '新增20题是开发期原创合成挑战题，未经独立医学审题；引用与supported来自同一Reasoner审核。','']
    for name in ('all50','original30','extension20','extension_anemia','extension_renal','extension_thyroid'):
        target=ids if name=='all50' else slices[name]
        lines += [f'## {name}（每条件{len(target)}题）','',
                  '| 条件 | 完成 | 答对 | 总Token | 模型调用 | 检索次数 | 返回调查 | 实际引用记忆的题 | 零检索且需求均获支持的记忆题 | 失败题 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for c,label in zip(CONDITIONS,LABELS):
            m=summaries[name][c];subset=[r for r in rows[c] if r['question_id'] in target]
            values=[label,len(subset),sum(bool(r.get('correct')) for r in subset),
                    *[m.get(k,0) for k in ('total_tokens','model_calls','retrieval_count','followup_rounds')],
                    sum(bool(r.get('reused_memory_ids')) for r in subset),m.get('memory_only_supported_tasks',0),
                    m.get('runtime_failed_questions',0)]
            lines.append('| '+' | '.join(map(str,values))+' |')
        lines.append('')
        if all([r['question_id'] for r in rows[c] if r['question_id'] in target]==target for c in CONDITIONS):
            for base,dest,label in [('text_off','structured_off','无记忆：结构化相对文本'),
                ('text_on','structured_on','有记忆：结构化相对文本'),
                ('structured_off','structured_on','结构化：启用记忆'),('text_off','text_on','文本：启用记忆')]:
                a,b=summaries[name][base],summaries[name][dest]
                if a['token_usage_complete'] and b['token_usage_complete'] and a['total_tokens']:
                    lines.append(f"- {label}：总Token变化{b['total_tokens']/a['total_tokens']-1:+.2%}，正确率变化{(b['accuracy']-a['accuracy'])*100:+.2f}个百分点。")
            lines.append('')
    lines+=['## 答错、未回答与运行问题','', '| 条件 | 题号 | 回答 | 状态 | 协议状态 |', '|---|---|---|---|---|']
    for c in CONDITIONS:
        for r in rows[c]:
            if not r.get('correct') or r.get('status') not in {'answered','ok'} or r.get('protocol_result',{}).get('status')=='issues':
                lines.append('| '+' | '.join(str(v) for v in [c,r['question_id'],r.get('answer'),r.get('status'),r.get('protocol_result',{}).get('status')])+' |')
    checkpoint=run/'checkpoint.sqlite3'
    if checkpoint.exists():
        with sqlite3.connect(checkpoint.resolve().as_uri()+'?mode=ro',uri=True) as db:
            calls=db.execute('SELECT condition,state,count(*) FROM calls GROUP BY condition,state').fetchall()
        lines+=['','账本调用状态：'+json.dumps(calls,ensure_ascii=False),'',
            '运行中的题目可能已有已保存调用但尚未出现在完成题汇总；未知usage不按零计费。',
            '通信和记忆必须分别比较；更少Token不自动证明更准确或证据更好。']
    atomic(run/'progress.md','\n'.join(lines)+'\n')
    atomic(run/'sliced-report.json',json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
    if (run/'report.json').exists():atomic(run/'summary.md',detailed_summary(run))
    if complete:
        finish(run,rows,summaries)
    return complete


def finish(run,rows,summaries):
    """Emit final comparisons only after identical complete streams exist."""
    lines=['# 50题正式API实验结论','',
        '四条件各50题，合计200/200任务完成。完成包含失败与未回答，不等于200题均答对或协议均有效。','',
        '| 条件 | 原30答对 | 新20答对 | 总Token | 检索次数 | 记忆片段引用题数 | 运行失败题 | 引用/协议问题题 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c,label in zip(CONDITIONS,LABELS):
        m=summaries['all50'][c]
        values=[label,round((summaries['original30'][c]['accuracy'] or 0)*30),
            round((summaries['extension20'][c]['accuracy'] or 0)*20),m['total_tokens'],m['retrieval_count'],
            sum(bool(r.get('reused_memory_ids')) for r in rows[c]),m['runtime_failed_questions'],
            m['protocol_status_counts'].get('issues',0)+m['protocol_status_counts'].get('invalid',0)]
        lines.append('| '+' | '.join(map(str,values))+' |')
    lines+=['','比较结论：','']
    for base,dest,label in [('text_off','structured_off','关闭记忆时结构化相对文本'),
            ('text_on','structured_on','开启记忆时结构化相对文本'),
            ('structured_off','structured_on','结构化路径开启记忆'),('text_off','text_on','文本路径开启记忆')]:
        a,b=summaries['all50'][base],summaries['all50'][dest]
        if not a['token_usage_complete'] or not b['token_usage_complete']:
            lines.append(f'- {label}：usage不完整，不能给出可靠Token收益比例。');continue
        delta=b['total_tokens']/a['total_tokens']-1
        lines.append(f"- {label}：总Token{delta:+.2%}；正确率变化{(b['accuracy']-a['accuracy'])*100:+.2f}个百分点；检索次数变化{b['retrieval_count']-a['retrieval_count']:+d}。")
    lines+=['','记忆是否实际进入计算：','',
        '| 条件 | 知识需求 | 首轮记忆暂定覆盖 | 最终记忆片段支持需求 | 零检索且引用记忆、需求均获支持的题 | 返回调查 | 需求审核错误 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for c,label in zip(CONDITIONS[2:],LABELS[2:]):
        m=summaries['all50'][c]
        lines.append('| '+' | '.join(map(str,[label,*[m.get(k,0) for k in ('knowledge_needs','memory_covered_needs',
            'memory_supported_needs','memory_only_supported_tasks','followup_rounds','needs_review_errors')]]))+' |')
        off=summaries['all50'][c.replace('_on','_off')]
        if m['total_tokens']>=off['total_tokens']:
            lines+=['',f'{label}：本批次没有观察到总Token净节省。局部命中或零检索不能抵消整体记忆展示、覆盖判断和补查输出时，不应宣称记忆优化有效。','']
        else:
            lines+=['',f'{label}：本批次观察到总Token降低，仍须结合准确率、协议错误及新增20题切片解释；不能把单次观察归为稳定因果收益。','']
    if all(summaries['extension20'][c]['accuracy']==1 for c in CONDITIONS):
        lines+=['新增20题仍全部答对：挑战题没有打破该模型的选项准确率天花板。保留这一结果，不按结果挑题或人为制造错误。','']
    lines+=['复用按存储原文区间匹配计数，独立于来源级关联；supported仍来自同一Reasoner，不代表独立医学认证。',
        '结构化比较包含ACP和角色视图组合，不能单独归因数字编码。问题/选项仍需进入独立API请求，未共享模型内部理解。',
        '新增20题为开发期原创合成题，未独立医学审题；连续性是知识关联与复用机会，病例均自足。',
        '早期检查批次另保存28次调用、53872Token；不并入正式条件对照，但属于本次开发实际API用量。','',
        '逐题与切片：progress.md、sliced-report.json、question-metrics.csv；完整机制指标：summary.md；原始账本：checkpoint.sqlite3。']
    atomic(run/'final-analysis.md','\n'.join(lines)+'\n')
    buffer=io.StringIO();writer=csv.writer(buffer)
    writer.writerow(['condition','question_id','correct','status','protocol_status','total_tokens','model_calls',
                     'retrieval_count','followup_rounds','memory_used','memory_supported_needs'])
    for c in CONDITIONS:
        for r in rows[c]:
            writer.writerow([c,r['question_id'],r.get('correct'),r['status'],r.get('protocol_result',{}).get('status'),
                *[r['metrics'].get(k,0) for k in ('total_tokens','model_calls','retrieval_count','followup_rounds','memory_used','memory_supported_needs')]])
    atomic(run/'question-metrics.csv',buffer.getvalue())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--pid',type=int)
    p.add_argument('--watch',action='store_true',help='Refresh until all four conditions finish')
    p.add_argument('--interval',type=int,default=30);args=p.parse_args()
    while True:
        complete=False
        try:
            if (args.run/'manifest.json').exists():
                complete=refresh(args.run)
                if complete:break
        except (FileNotFoundError,json.JSONDecodeError,sqlite3.OperationalError) as exc:
            print(type(exc).__name__,flush=True)
        if not args.watch and (not args.pid or not alive(args.pid)):
            if args.pid and args.run.exists():
                atomic(args.run/'runner-status.json',json.dumps({'runner_alive':False,'complete':complete,
                    'note':'Runner stopped. If incomplete, inspect checkpoint and log; do not infer full-test results.'},indent=2)+'\n')
            break
        time.sleep(args.interval)
