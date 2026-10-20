"""Validate the localhost model, JSON mode and actual token usage, once."""
import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import Request,build_opener,ProxyHandler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',default='http://127.0.0.1:8010/v1')
    p.add_argument('--output',type=Path,default=Path('experiments/validation/qwen32b-local-20261020/api-probe.json'))
    p.add_argument('--model-name',default='qwen2.5-32b-instruct-int8')
    args=p.parse_args()
    if urlparse(args.base_url).hostname not in {'127.0.0.1','localhost','::1'}:
        raise ValueError('此探针只允许本机地址，不调用云模型')
    headers={'Authorization':'Bearer local-moreagent','Content-Type':'application/json'}
    opener=build_opener(ProxyHandler({}))
    def call(path,data=None):
        req=Request(args.base_url.rstrip('/')+path,headers=headers,
                    data=json.dumps(data).encode() if data is not None else None)
        with opener.open(req,timeout=180) as response:return json.load(response)
    models=call('/models')
    expected=args.model_name
    if expected not in {m['id'] for m in models['data']}:raise ValueError('服务模型名不匹配')
    started=time.monotonic()
    body={'model':expected,'temperature':0,'max_tokens':64,
        'response_format':{'type':'json_object'},'messages':[
            {'role':'system','content':'只输出一个JSON对象，不要解释。'},
            {'role':'user','content':'请返回JSON对象：{"ok":true}'}]}
    if expected.lower().rsplit('/',1)[-1].startswith('qwen3-'):
        body['chat_template_kwargs']={'enable_thinking':False}
    response=call('/chat/completions',body)
    parsed=json.loads(response['choices'][0]['message']['content'])
    if parsed!={'ok':True}:raise ValueError('JSON探针内容不符合预期')
    usage=response['usage']
    if not (usage['prompt_tokens']>0 and usage['completion_tokens']>0
            and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']):
        raise ValueError('模型usage缺失或不一致')
    result={'model':response['model'],'json_object_valid':True,'output':parsed,'usage':usage,
            'finish_reason':response['choices'][0]['finish_reason'],'elapsed_seconds':time.monotonic()-started,
            'generation_requests':1,'cloud_requests':0}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
