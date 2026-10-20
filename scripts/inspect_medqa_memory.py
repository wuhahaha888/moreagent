"""Read-only inspection of shared-memory metadata; no model or embedding load."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


def inspect(path, limit):
    if not path.is_file():
        raise ValueError('记忆数据库不存在')
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute('SELECT id,source_agent,created_at,task_topic,summary,kind,status,content FROM memory ORDER BY created_at,id').fetchall()
    result=[]
    for row in rows[:limit]:
        item=dict(row)
        content=json.loads(item.pop('content'))
        item['created_at_utc']=datetime.fromtimestamp(item['created_at'],timezone.utc).isoformat()
        item['validation']=content.get('validation','legacy')
        item['task_group']=content.get('task_group')
        result.append(item)
    return {'database':str(path.resolve()),'records':len(rows),
            'counts_by_kind':dict(Counter(r['kind'] for r in rows)),
            'required_metadata_present':all(all(r[k] is not None and r[k]!='' for k in ['id','source_agent','created_at','task_topic','summary']) for r in rows),
            'items':result,'note':'curated表示已选入记忆，不代表独立医学认证。'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('database',type=Path)
    parser.add_argument('--limit',type=int,default=10)
    args=parser.parse_args()
    if args.limit<1:parser.error('--limit必须大于0')
    print(json.dumps(inspect(args.database,args.limit),ensure_ascii=False,indent=2))
