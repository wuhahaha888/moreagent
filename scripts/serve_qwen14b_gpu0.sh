#!/usr/bin/env bash
# Qwen3-14B原始BF16，GPU0；非思考由客户端显式关闭。
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
QWEN_PYTHON=${QWEN_LOCAL_PYTHON:-$PROJECT_ROOT/.venv-qwen-local/bin/python}
QWEN_VLLM_BASE=${QWEN_VLLM_BASE:-$PROJECT_ROOT/../miniconda3/envs/medrgag}
QWEN_PORT=${QWEN_LOCAL_PORT:-8010}
QWEN_MODEL_DIR=${QWEN_LOCAL_MODEL_DIR:-/data/SharedFile/Qwen/Qwen3-14B}
QWEN_MAX_SEQS=${QWEN_LOCAL_MAX_SEQS:-4}
cd -- "$PROJECT_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$QWEN_VLLM_BASE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
"$QWEN_PYTHON" -B - "$QWEN_MODEL_DIR" "$QWEN_PORT" "$QWEN_MAX_SEQS" <<'PY'
import json,socket,sys
from pathlib import Path
import torch
root=Path(sys.argv[1]);config=json.loads((root/'config.json').read_text())
if config.get('model_type')!='qwen3' or config.get('num_hidden_layers')!=40 or config.get('quantization_config'):
    raise SystemExit('目标须为原始Qwen3-14B BF16；拒绝静默切换其他模型或精度')
index=json.loads((root/'model.safetensors.index.json').read_text())
if not all((root/f).is_file() for f in set(index['weight_map'].values())):
    raise SystemExit('模型分片不完整')
if not 1<=int(sys.argv[3])<=4:raise SystemExit('并发保护范围为1至4')
with socket.socket() as sock:sock.bind(('127.0.0.1',int(sys.argv[2])))
if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise SystemExit('GPU0不可用')
if torch.cuda.mem_get_info()[0]<36*1024**3:raise SystemExit('GPU0空闲显存不足36GiB，保留已有进程，不启动')
PY
exec "$QWEN_PYTHON" -B -m vllm.entrypoints.openai.api_server \
  --model "$QWEN_MODEL_DIR" --served-model-name qwen3-14b-bf16 \
  --host 127.0.0.1 --port "$QWEN_PORT" --api-key local-moreagent \
  --dtype bfloat16 --tensor-parallel-size 1 --gpu-memory-utilization 0.82 \
  --max-model-len 8192 --max-num-seqs "$QWEN_MAX_SEQS" --max-num-batched-tokens 8192 \
  --enable-prefix-caching --enable-chunked-prefill --generation-config vllm \
  --guided-decoding-backend xgrammar --disable-log-requests
