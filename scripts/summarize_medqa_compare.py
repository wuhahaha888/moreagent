#!/usr/bin/env python3
"""只读汇总四组实验；可等待指定实验进程结束后生成 Markdown。"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import time


CONDITIONS = ('text_off', 'structured_off', 'text_on', 'structured_on')
LABELS = ('纯文本·无记忆', '结构化·无记忆', '纯文本·有记忆', '结构化·有记忆')


def alive(pid):
    try:
        state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
        return state not in {'Z', 'X'}
    except FileNotFoundError:
        return False


def summarize(run):
    report = json.loads((run / 'report.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    expected = len(manifest['question_ids'])
    rows = {c: [json.loads(line) for line in (run / f'{c}.jsonl').read_text().splitlines() if line.strip()]
            if (run / f'{c}.jsonl').exists() else [] for c in CONDITIONS}
    ids = manifest['question_ids']
    complete = all([r['question_id'] for r in rows[c]] == ids for c in CONDITIONS)
    lines = ['# 本地四组实验汇总', '',
             '生成时间：' + datetime.now(timezone.utc).isoformat(), '',
             f'状态：{"四组全量完成" if complete else "尚未完成全量，以下仅为已完成部分"}；每组目标 {expected} 题。', '',
             '| 条件 | 完成 | 答对 | 正确率 | 输入 Token | 输出 Token | 总 Token | LLM 调用 | 平均秒/题 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c, label in zip(CONDITIONS, LABELS):
        m = report.get(c, {})
        n = m.get('questions_completed', 0)
        correct = sum(bool(r.get('correct')) for r in rows[c])
        accuracy = f'{correct / n:.2%}' if n else '—'
        duration = f"{m['mean_seconds']:.2f}" if n else '—'
        values = [label, n, correct, accuracy, *[m.get(k, 0) for k in
                  ('prompt_tokens', 'completion_tokens', 'total_tokens', 'model_calls')], duration]
        lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', '| 条件 | 检索次数 | 补查轮数 | 契约纠错调用 | 运行失败题 | 协议有问题题 | usage 完整 | 协作字节¹ | 非文本次数/字节 |',
              '|---|---:|---:|---:|---:|---:|---|---:|---|']
    for c, label in zip(CONDITIONS, LABELS):
        m = report.get(c, {})
        values = [label, *[m.get(k, 0) for k in ('retrieval_count', 'followup_rounds', 'contract_repairs', 'runtime_failed_questions')],
                  m.get('protocol_status_counts', {}).get('issues', 0), m.get('token_usage_complete', False),
                  m.get('request_bytes', 0) + m.get('response_bytes', 0),
                  f"{m.get('state_transfers', 0)}/{m.get('state_bytes', 0)}"]
        lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', '¹协作字节为 request_bytes + response_bytes，不等于网络抓包字节；系统提示和协议控制字节另有指标。协议有问题可能涉及引用，不等于 JSON 解析失败。', '',
              '| 记忆条件 | 实际进入角色输入条数² | 实际策略执行条数 | 最终记忆证据引用条数 | 写入请求数³ | 无教材检索的复用题 |',
              '|---|---:|---:|---:|---:|---:|']
    for c, label in zip(CONDITIONS[2:], LABELS[2:]):
        m = report.get(c, {})
        funnel = m.get('memory_funnel', {})
        values = [label, sum(funnel.get('presented', {}).values()),
                  funnel.get('executed', {}).get('strategy', 0), funnel.get('cited', {}).get('evidence', 0),
                  m.get('memory_written', 0), m.get('reused_without_search_questions', 0)]
        lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', '²跨题累计事件计数，不代表全局唯一记忆数。³写入请求可包含幂等去重，不等于新增唯一记忆数。候选召回、实际执行、最终引用分别报告；引用也不自动证明复用正确。', '']
    if manifest.get('workflow') == 'need_first':
        lines += ['需求驱动流程：先规划，再按需求召回记忆；Investigator判断暂定覆盖，Reasoner核验并可返回调查。两种通信模式使用同一流程。', '',
                  '| 条件 | 知识需求数 | 首轮记忆暂定覆盖需求 | 最终引用记忆片段的支持需求 | 零检索且引用记忆、需求均获支持的题 | 已覆盖查询拦截 | 需求审核错误 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for c,label in zip(CONDITIONS,LABELS):
            m=report.get(c,{})
            values=[label,*[m.get(k,0) for k in ('knowledge_needs','memory_covered_needs','memory_supported_needs',
                    'memory_only_supported_tasks','queries_skipped_covered','needs_review_errors')]]
            lines.append('| '+' | '.join(map(str,values))+' |')
        lines += ['', '“支持”来自同一Reasoner的需求审核及程序原文/可见性检查，不是独立医学认证；上述计数不证明净Token收益。净收益仍以同流程有/无记忆的完整API usage对照计算。', '']
    if manifest.get('communication_version')=='moreagent.handoff/1':
        lines+=['通信路径版本：moreagent.handoff/1。Text 使用自然语言交接，structured 使用 ACP-1 进程内二进制派发。', '',
            '| 条件 | ACP帧数 | 请求/结果字节 | 握手字节 | 登记内容序列化字节 | 证据引用回读 | 向量句柄消费 | 自然语言交接字符 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for c,label in zip(CONDITIONS,LABELS):
            h=report.get(c,{}).get('handoff_metrics',{})
            values=[label,h.get('acp_frames',0),h.get('acp_request_bytes',0)+h.get('acp_result_bytes',0),
                h.get('acp_handshake_bytes',0),h.get('acp_object_registration_bytes',0),
                h.get('acp_evidence_ref_reads',0),h.get('state_handle_reads',0),h.get('natural_handoff_chars',0)]
            lines.append('| '+' | '.join(map(str,values))+' |')
        lines+=['', 'ACP字节与模型提示正文的UTF-8字节分别统计；引用对象保存在任务内对象表，登记字节是内容序列化大小，不是网络抓包流量。握手和向量句柄不直接折算聊天Token。', '']
    if manifest.get('workspace_policy') == 'role_views_v1':
        lines += ['共享状态策略：role_views_v1，仅structured启用；Text保留原基线。比较的是协议与上下文组合方案，不能只归因于数字编码。', '',
                  '| 条件 | 角色视图次数 | 展开前字节 | 实际视图字节 | 原文条件缩略次数 | 已提交角色变更 |',
                  '|---|---:|---:|---:|---:|---:|']
        for c, label in zip(CONDITIONS, LABELS):
            w = report.get(c, {}).get('workspace_metrics', {})
            values = [label, *[w.get(k, 0) for k in ('workspace_views', 'workspace_input_bytes_before',
                      'workspace_input_bytes_after', 'workspace_brief_views', 'workspace_patch_count')]]
            lines.append('| ' + ' | '.join(map(str, values)) + ' |')
        lines += ['', '展开前后字节是同一次任务的输入投影视图统计；不能替代实际API token或当作独立质量评测。原题、全部选项及必要证据仍按角色保留。', '']
    if complete:
        for label, base, target in [('无记忆时结构化相对纯文本', 'text_off', 'structured_off'),
                                    ('有记忆时结构化相对纯文本', 'text_on', 'structured_on'),
                                    ('纯文本加入记忆', 'text_off', 'text_on'),
                                    ('结构化加入记忆', 'structured_off', 'structured_on')]:
            a, b = report[base], report[target]
            delta = b['accuracy'] - a['accuracy']
            cost = (f"总 Token 变化 {b['total_tokens'] / a['total_tokens'] - 1:+.2%}"
                    if a['token_usage_complete'] and b['token_usage_complete'] and a['total_tokens'] else '存在未知 usage，不比较 Token 比例')
            lines.append(f"- {label}：正确率变化 {delta * 100:+.2f} 个百分点；{cost}；LLM 调用变化 {b['model_calls'] - a['model_calls']:+d} 次。")
    else:
        lines.append('四组尚未齐全，不计算跨组收益。')
    lines += ['', '以上是当前样本的观察差异；须结合错误题、未回答题和有效复用情况解释，不能仅凭 Token 下降宣称优化成功。', '',
              '统计边界：report 按完成题更新；执行中调用先保存在 checkpoint。未返回 usage 的调用无法精确计量。Embedding 与程序计算成本不计入聊天 Token。本地与旧 API tokenizer 不同，不把跨模型 Token 变化直接归因为通信机制。', '',
              '原始结果：同目录 report.json、comparison.json、各条件 jsonl、checkpoint.sqlite3；本文件未改变实验配置或源代码。', '']
    checkpoint = run / 'checkpoint.sqlite3'
    if checkpoint.exists():
        connection = sqlite3.connect(checkpoint.resolve().as_uri() + '?mode=ro', uri=True)
        try:
            saved = [json.loads(row[0]) for row in connection.execute("SELECT usage FROM calls WHERE state='complete'")]
            unknown = connection.execute("SELECT count(*) FROM calls WHERE state!='complete'").fetchone()[0]
        finally:
            connection.close()
        known = [u for u in saved if u.get('source') == 'model' and u.get('usage_complete')]
        lines += [f"账本核对：已保存响应 {len(saved)} 次，其中真实 usage 完整 {len(known)} 次，已知总 Token {sum(u['total_tokens'] for u in known)}；未保存响应的请求记录 {unknown} 次。包含未完成题目的已保存调用，可能高于按完成题汇总的 report；未知请求不当作零成本。", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.wait_pid:
        while alive(args.wait_pid):
            time.sleep(30)
    output = args.output or args.run / 'summary.md'
    text = summarize(args.run)
    temporary = output.with_name(output.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, output)
    print(output, flush=True)


if __name__ == '__main__':
    main()
