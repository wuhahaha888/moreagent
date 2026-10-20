#!/usr/bin/env python3
"""只读显示四组评测进度；不调用模型、不改检查点、不启动或停止任务。"""
import argparse
from contextlib import closing
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import time

CONDITIONS = ('text_off', 'structured_off', 'text_on', 'structured_on')
LABELS = ('纯文本/无记忆', '结构化/无记忆', '纯文本/有记忆', '结构化/有记忆')


def inspect_condition(folder):
    checkpoint = folder / 'checkpoint.sqlite3'
    if not checkpoint.exists():
        return {'completed': 0, 'stage': '尚未建立检查点', 'known_tokens': 0, 'calls': 0,
                'pending': 0, 'failed': 0, 'current': '—', 'last_status': '—'}
    with closing(sqlite3.connect(checkpoint.resolve().as_uri() + '?mode=ro', uri=True, timeout=.2)) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')  # 多条查询取同一快照；退出时关闭事务。
        completed = db.execute("SELECT count(*) FROM questions WHERE state='complete'").fetchone()[0]
        current = db.execute("SELECT position,question_id,state FROM questions WHERE state!='complete' ORDER BY position LIMIT 1").fetchone()
        last = db.execute("SELECT result FROM questions WHERE state='complete' ORDER BY position DESC LIMIT 1").fetchone()
        calls = db.execute('SELECT state,usage FROM calls').fetchall()
        failures = db.execute("SELECT count(*) FROM questions WHERE state='complete' AND json_extract(result,'$.answer') IS NULL").fetchone()[0]
        stage = '本组已结束或等待调度'
        if current:
            call = db.execute('SELECT role,state FROM calls WHERE position=? ORDER BY ordinal DESC,attempt DESC LIMIT 1',
                              (current['position'],)).fetchone()
            if current['state'] == 'ready':
                stage = '答案已保存，提交记忆中'
            elif call and call['state'] == 'pending':
                stage = '等待 ' + call['role'] + ' 返回'
            elif call:
                stage = call['role'] + ' 已返回，程序处理中'
            else:
                stage = '题目初始化/记忆预取'
        usage = [json.loads(c['usage']) for c in calls if c['state'] == 'complete' and c['usage']]
        known = [u for u in usage if u.get('source') == 'model' and u.get('usage_complete')]
        return {'completed': completed, 'stage': stage,
                'known_tokens': sum(u['total_tokens'] for u in known), 'calls': len(usage),
                'pending': sum(c['state'] != 'complete' for c in calls), 'failed': failures,
                'current': str(current['position'] + 1) if current else '—',
                'question_id': current['question_id'] if current else None,
                'last_status': json.loads(last['result']).get('status', 'unknown') if last else '—'}


def render(run):
    path = run / 'run-status.json'
    status = json.loads(path.read_text()) if path.is_file() else {}
    if not status and not (run / 'conditions').is_dir():
        raise ValueError('未找到并行实验目录：' + str(run))
    target = status.get('target_per_condition', '?')
    lines = ['更新时间：' + datetime.now().astimezone().isoformat(timespec='seconds'),
             '实验目录：' + str(run.resolve()),
             '条件             完成       当前题   已返回调用   已知Token   无答案题   当前阶段']
    details, counts = [], []
    for condition, label in zip(CONDITIONS, LABELS):
        try:
            info = inspect_condition(run / 'conditions' / condition)
            counts.append(info['completed'])
            lines.append(f"{label:<13} {str(info['completed']) + '/' + str(target):<10} {info['current']:<8} "
                         f"{info['calls']:<12} {info['known_tokens']:<11} {info['failed']:<10} {info['stage']}")
            if info.get('question_id'):
                details.append(f"  {condition} 当前题ID：{info['question_id']}")
        except (sqlite3.Error, OSError, ValueError) as error:
            lines.append(label + '：暂时无法读取（' + type(error).__name__ + '），下次刷新重试')
    total = str(target * 4) if isinstance(target, int) else '?'
    lines += ['', f'四组合计已完成：{sum(counts)}/{total} 次任务运行',
              '最近调度状态：' + str(status.get('state', 'unknown')),
              '子进程PID：' + json.dumps(status.get('active_pids', {}), ensure_ascii=False)]
    if 'updated_unix' in status:
        age = max(0, time.time() - status['updated_unix'])
        lines.append(f'调度状态最后刷新距今：{age:.0f} 秒（持续增长时检查主进程与日志）')
    lines += ['', *details, '',
              '“等待Agent返回”包含服务端排队与生成；已知Token含当前题已落盘的调用，不含尚未返回的请求。',
              '完成数含失败题；全部跑完后还需检查无答案题、运行错误和usage完整性。Ctrl+C仅退出监视。']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not args.interval >= 1:
        parser.error('刷新间隔至少1秒')
    try:
        while True:
            text = render(args.run)
            if not args.once and os.isatty(1):
                print('\033[2J\033[H', end='')
            print(text, flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
