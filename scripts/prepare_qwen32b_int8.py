"""Prepare a local INT8 copy on the one explicitly visible GPU; never train."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('/data/SharedFile/Qwen/Qwen2___5-32B-Instruct'))
    parser.add_argument('--output',type=Path,default=root/'models/Qwen2.5-32B-Instruct-bnb-int8')
    args=parser.parse_args()
    source=args.source.resolve();target=args.output.resolve()
    config=source/'config.json';index=source/'model.safetensors.index.json'
    fingerprint={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                 for name in ('config.json','model.safetensors.index.json','tokenizer_config.json')}
    files=set(json.loads(index.read_text())['weight_map'].values())
    fingerprint['weight_files']={name:{'bytes':(source/name).stat().st_size,
                                      'mtime_ns':(source/name).stat().st_mtime_ns} for name in sorted(files)}
    marker=target/'moreagent-deployment.json'
    if marker.is_file():
        recorded=json.loads(marker.read_text())
        if recorded['source_fingerprint']!=fingerprint:raise ValueError('原始权重元数据变化，不能复用旧量化副本')
        weights=json.loads((target/'model.safetensors.index.json').read_text())['weight_map'].values()
        if not all((target/name).is_file() for name in weights):raise ValueError('量化副本文件缺失')
        print('已存在完整INT8副本:',target,flush=True);return
    if target.exists():raise ValueError('目标目录已存在但缺少完成标记；使用新目录，不覆盖未知产物')
    if os.getenv('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('此脚本要求CUDA_VISIBLE_DEVICES=0，防止误用其他GPU')
    import torch
    import importlib.metadata
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('未发现唯一可用的GPU0')
    free,total=torch.cuda.mem_get_info()
    if free<38*1024**3:raise RuntimeError('GPU0空闲显存低于38GiB，不开始32B INT8转换')
    print(json.dumps({'gpu':torch.cuda.get_device_name(0),'free_gib':free/1024**3,
                      'source':str(source),'precision':'bitsandbytes-int8'},ensure_ascii=False),flush=True)
    temporary=target.with_name(target.name+'.preparing')
    temporary.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained(str(source),local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(str(source),local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    model.eval()
    print('INT8加载完成，开始保存派生副本',flush=True)
    model.save_pretrained(str(temporary),safe_serialization=True,max_shard_size='4GB')
    tokenizer.save_pretrained(str(temporary))
    deployment={'model_name':'qwen2.5-32b-instruct-int8','source':str(source),
        'source_fingerprint':fingerprint,'precision':'bitsandbytes-int8',
        'non_quantized_dtype':'bfloat16','gpu':torch.cuda.get_device_name(0),
        'memory_footprint_bytes':model.get_memory_footprint(),'prepare_seconds':time.monotonic()-started,
        'packages':{k:importlib.metadata.version(k) for k in ('torch','transformers','bitsandbytes','accelerate')}}
    (temporary/'moreagent-deployment.json').write_text(json.dumps(deployment,ensure_ascii=False,indent=2)+'\n')
    temporary.rename(target)
    print('准备完成:',target,flush=True)


if __name__=='__main__':main()
