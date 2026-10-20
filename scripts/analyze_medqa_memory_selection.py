"""Read saved runs and compare memory policies without model calls or mutation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def load(label,root):
    manifest=json.loads((root/'manifest.json').read_text())
    report=json.loads((root/'report.json').read_text())
    if len(report)!=1:raise ValueError('每个输入目录应只含一个run条件')
    condition,stats=next(iter(report.items()))
    rows=[json.loads(line) for line in (root/(condition+'.jsonl')).read_text().splitlines() if line.strip()]
    searches=[h for r in rows for h in r['retrieval_history']]
    counts=Counter(' '.join(h['query'].casefold().split()) for h in searches)
    source_counts=Counter(i for h in searches for i in h.get('evidence_ids',[]))
    per_question=[]
    for r in rows:
        planner=next((e for e in r['events'] if e.get('agent')=='Planner'),{})
        per_question.append({'question_id':r['question_id'],'correct':r.get('correct'),'tokens':r['metrics']['total_tokens'],
            'searches':r['metrics']['retrieval_count'],'seconds':r['metrics']['elapsed_seconds'],
            'funnel':r['metrics'].get('memory_funnel'),
            'planner_preview_chars':len(json.dumps(planner.get('input',{}).get('memory_hits',[]),ensure_ascii=False,separators=(',',':'))),
            'selection':[e for e in r['events'] if e.get('tool')=='memory_selection'],
            'strategy_executions':[e for e in r['events'] if e.get('tool')=='memory_strategy_execution']})
    return {'label':label,'path':str(root.resolve()),'condition':condition,'question_ids':[r['question_id'] for r in rows],
        'manifest':manifest,'stats':stats,'per_question':per_question,
        'repeated_exact_queries':sum(n-1 for n in counts.values()),
        'repeated_returned_source_ids':sum(n-1 for n in source_counts.values()),
        'note':'重复来源不必然是无效工作；策略执行归因是已核对ID的Agent声明，不是因果证明。'}


def compare(runs):
    first=runs[0]
    fields=('dataset_sha256','textbooks_version','question_ids','model','source_sha256')
    issues=[]
    for run in runs:
        for field in fields:
            if run['manifest'].get(field)!=first['manifest'].get(field):issues.append(run['label']+': '+field+' mismatch')
        for key in ('max_calls','max_searches','max_reads','memory_threshold','top_k','candidates','evidence_processor','max_output_tokens'):
            if run['manifest']['settings'].get(key)!=first['manifest']['settings'].get(key):issues.append(run['label']+': '+key+' mismatch')
        if run['question_ids']!=first['manifest']['question_ids']:issues.append(run['label']+': incomplete or reordered questions')
        if not run['stats'].get('token_usage_complete'):issues.append(run['label']+': incomplete provider usage')
        if run['stats'].get('runtime_failed_questions'):issues.append(run['label']+': runtime failures')
        if run['stats'].get('resume_count'):issues.append(run['label']+': resumed latency not comparable')
    reference=next((r for r in runs if r['label']=='off'),first)
    for run in runs:
        r,b=run['stats'],reference['stats']
        run['relative_to_'+reference['label']]={key:(r[key]/b[key]-1) if b.get(key) else None
            for key in ['total_tokens','retrieval_count','mean_seconds','communication_chars']}
        run.pop('manifest')
    return {'matched_by_recorded_controls':not issues,'issues':issues,'runs':runs,
            'model_calls_during_analysis':0,'note':'单次开发小样本；新旧方案同时改变候选选择和角色提示。时延及服务缓存仍可能波动，不是正式泛化或显著性结论。'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',action='append',required=True,help='label=结果目录，例如off=experiments/results/...')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    runs=[]
    for spec in args.run:
        label,root=spec.split('=',1)
        runs.append(load(label,Path(root)))
    result=compare(runs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'matched_by_recorded_controls':result['matched_by_recorded_controls'],
        'issues':result['issues'],'output':str(args.output)},ensure_ascii=False))


if __name__=='__main__':main()
