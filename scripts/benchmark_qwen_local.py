#!/usr/bin/env python3
"""固定长度的本地 JSON 推理测速；请求成本单独记录，不混入 QA 实验。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import Request, build_opener, ProxyHandler


def benchmark(base_url, concurrency, requests, tokens, model_name='qwen2.5-32b-instruct-int8'):
    if urlparse(base_url).hostname not in {'127.0.0.1', 'localhost', '::1'}:
        raise ValueError('只允许本机测速')
    def call(index):
        body = {'model': model_name, 'temperature': 0,
                'max_tokens': tokens, 'response_format': {'type': 'json_object'},
                'messages': [{'role': 'system', 'content': '只输出JSON对象。'},
                             {'role': 'user', 'content': '输出JSON对象，values是从1到1000的整数数组，逐项写出，不要省略。'}]}
        if model_name.lower().rsplit('/',1)[-1].startswith('qwen3-'):
            body['chat_template_kwargs']={'enable_thinking':False}
        req = Request(base_url.rstrip('/') + '/chat/completions',
                      data=json.dumps(body).encode(), headers={
                          'Authorization': 'Bearer local-moreagent', 'Content-Type': 'application/json'})
        started = time.monotonic()
        with build_opener(ProxyHandler({})).open(req, timeout=600) as response:
            result = json.load(response)
        usage = result['usage']
        assert usage['prompt_tokens'] + usage['completion_tokens'] == usage['total_tokens']
        return {'index': index, 'seconds': time.monotonic() - started,
                'usage': usage, 'finish_reason': result['choices'][0]['finish_reason']}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        calls = list(pool.map(call, range(requests)))
    seconds = time.monotonic() - started
    output = sum(c['usage']['completion_tokens'] for c in calls)
    return {'model':model_name,'concurrency': concurrency, 'requests': requests, 'seconds': seconds,
            'output_tokens': output, 'output_tokens_per_second': output / seconds,
            'total_tokens': sum(c['usage']['total_tokens'] for c in calls), 'calls': calls,
            'note': '合成固定长度JSON输出吞吐；不是QA正确率，截断为预期行为；不计入实验成本。'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8010/v1')
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--model-name',default='qwen2.5-32b-instruct-int8')
    parser.add_argument('--requests', type=int, default=4)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.concurrency, args.requests, args.tokens) < 1:
        parser.error('参数必须为正')
    result = benchmark(args.base_url, args.concurrency, args.requests, args.tokens,args.model_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'calls'}, ensure_ascii=False), flush=True)
