#!/usr/bin/env python3
"""Export immutable, auditable result snapshots without touching live runs.

SQLite's online backup gives each database a consistent read snapshot. Separate
memory/checkpoint databases are NOT a coordinated resume bundle. No API calls.
"""
import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from LangChain.medqa.cli import summarize, compare_reports

CONDITIONS=('text_off','structured_off','text_on','structured_on')


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def compress(source,target):
    with source.open('rb') as src,target.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as out:
            shutil.copyfileobj(src,out)


def snapshot_database(source,target):
    with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True,timeout=30) as src:
        with sqlite3.connect(target) as dst:
            src.backup(dst,pages=1024,sleep=.05)
            if dst.execute('PRAGMA quick_check').fetchone()[0]!='ok':
                raise ValueError('Snapshot integrity check failed: '+source.name)


def export_run(run,out,include_databases):
    out.mkdir()
    captured=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    with tempfile.TemporaryDirectory(prefix='medqa-export-') as temporary:
        db_path=Path(temporary)/'checkpoint.sqlite3'
        snapshot_database(run/'checkpoint.sqlite3',db_path)
        with sqlite3.connect(db_path) as db:
            meta={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
            manifest=meta['manifest'];ids=manifest['question_ids']
            rows={c:[json.loads(r[0]) for r in db.execute(
                "SELECT result FROM questions WHERE condition=? AND state='complete' ORDER BY position",(c,))]
                for c in CONDITIONS}
            calls=[json.loads(r[0]) for r in db.execute("SELECT usage FROM calls WHERE state='complete'")]
            call_states=dict(db.execute('SELECT state,count(*) FROM calls GROUP BY state'))
            task_states=[list(r) for r in db.execute('SELECT condition,state,count(*) FROM questions GROUP BY condition,state')]
        report={c:summarize(rows[c]) for c in CONDITIONS}
        completed={c:[r['question_id'] for r in rows[c]] for c in CONDITIONS}
        complete=all(v==ids for v in completed.values())
        write_json(out/'manifest.json',manifest)
        write_json(out/'report.json',report)
        write_json(out/'comparison.json',compare_reports(report,completed,ids))
        for c in CONDITIONS:
            if not rows[c]:continue
            raw=(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows[c])).encode()
            with (out/(c+'.jsonl.gz')).open('wb') as f:
                with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as g:g.write(raw)
        databases={}
        if include_databases:
            compress(db_path,out/'checkpoint.sqlite3.gz')
            databases['checkpoint.sqlite3']={'sha256_uncompressed':digest(db_path),'captured_at':captured}
            for source in sorted(run.glob('*.sqlite3')):
                if source.name=='checkpoint.sqlite3':continue
                copy=Path(temporary)/source.name
                snapshot_database(source,copy)
                compress(copy,out/(source.name+'.gz'))
                databases[source.name]={'sha256_uncompressed':digest(copy),
                    'captured_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
        state={'run_name':run.name,'captured_at':captured,'expected_per_condition':len(ids),
            'completed':{c:len(rows[c]) for c in CONDITIONS},'matched_complete':complete,
            'saved_response_count':len(calls),'call_states':call_states,'task_states':task_states,
            'known_saved_response_tokens':sum(u.get('total_tokens',0) for u in calls),
            'unknown_request_records':sum(v for k,v in call_states.items() if k!='complete'),
            'databases':databases,'purpose':'audit_snapshot; not a coordinated cross-database resume bundle',
            'source_run_untouched':True,'api_calls_during_export':0}
        write_json(out/'snapshot.json',state)
        title='完整题序快照（含失败题）' if complete else '部分结果快照，不能作为全量收益结论'
        lines=['# '+run.name,'',title+'。采集时间：'+captured,'',
            '| 条件 | 完成/目标 | 答对 | 总Token | 运行失败 | usage完整 |',
            '|---|---:|---:|---:|---:|---|']
        for c in CONDITIONS:
            m=report[c];values=[c,f'{len(rows[c])}/{len(ids)}',sum(bool(r.get('correct')) for r in rows[c]),
                m['total_tokens'],m['runtime_failed_questions'],m['token_usage_complete']]
            lines.append('| '+' | '.join(map(str,values))+' |')
        lines+=['','snapshot.json记录实际账本状态；运行中pending请求不能视为失败或零费用。',
            'report按完成题统计，账本还可能有未完成题的已保存调用；未知usage保留为未知。',
            'jsonl.gz含逐题完整轨迹；sqlite3.gz为各库独立只读备份，可解压审计，不作为直接续跑入口。',
            '原始实验目录、数据库、日志均保留本机，导出不停止运行，不改写结果。']
        (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return state,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,default=ROOT/'experiments/results')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--include-databases',action='store_true')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    states=[];buffer=io.StringIO();writer=csv.writer(buffer,lineterminator='\n')
    writer.writerow(['run','captured_at','matched_complete','condition','completed','correct','total_tokens',
                     'runtime_failed_questions','token_usage_complete'])
    for run in sorted(args.run_root.iterdir()):
        if not (run/'checkpoint.sqlite3').is_file():continue
        state,report=export_run(run,out/run.name,args.include_databases);states.append(state)
        for c,m in report.items():
            writer.writerow([run.name,state['captured_at'],state['matched_complete'],c,m['questions_completed'],
                round((m['accuracy'] or 0)*m['questions_completed']),m['total_tokens'],m['runtime_failed_questions'],m['token_usage_complete']])
    write_json(out/'index.json',states)
    (out/'summary.csv').write_text(buffer.getvalue(),encoding='utf-8')
    lines=['# 实验结果只读快照','',
        '保留完整与未完成批次。每个checkpoint使用SQLite在线备份采集；不同数据库分别采集，不保证跨库同一时刻，不能直接用作运行中恢复包。',
        '原始目录继续保留本机，运行中实验会继续变化；GitHub展示的是采集时刻的固定快照。','',
        '| 批次 | 完成任务/目标 | 全题序齐全 |','|---|---:|---|']
    for s in states:
        n=sum(s['completed'].values());target=s['expected_per_condition']*4
        lines.append(f"| [{s['run_name']}]({s['run_name']}/README.md) | {n}/{target} | {s['matched_complete']} |")
    lines+=['','完整题序不等于所有题运行成功。失败题、未知usage和负结果均保留，不将部分实验与不同版本拼接成全量收益。',
        'summary.csv汇总条件；各批次report.json、comparison.json、逐题jsonl.gz和可选数据库快照用于复核。',
        '解压数据库：gzip -dk checkpoint.sqlite3.gz；解压轨迹：gzip -dk text_on.jsonl.gz。建议复制到临时目录审计。']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(out/'checksums.json',{str(f.relative_to(out)):digest(f) for f in sorted(out.rglob('*')) if f.is_file()})
    print(json.dumps({'output':str(out),'runs':len(states),'api_calls':0,
        'completed':{s['run_name']:sum(s['completed'].values()) for s in states}}))


if __name__=='__main__':main()
