#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
QWEN_PYTHON=${QWEN_LOCAL_PYTHON:-$PROJECT_ROOT/.venv-qwen-local/bin/python}
QWEN_VLLM_BASE=${QWEN_VLLM_BASE:-$PROJECT_ROOT/../miniconda3/envs/medrgag}
QWEN_PORT=${QWEN_LOCAL_PORT:-8010}
QWEN_MODEL_DIR=${QWEN_LOCAL_MODEL_DIR:-$PROJECT_ROOT/models/Qwen2.5-32B-Instruct-bnb-int8}
QWEN_MAX_SEQS=${QWEN_LOCAL_MAX_SEQS:-4}
QWEN_BATCH_TOKENS=${QWEN_LOCAL_BATCH_TOKENS:-8192}
cd -- "$PROJECT_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$QWEN_VLLM_BASE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
if [[ ! -x "$QWEN_PYTHON" ]]; then
  printf '本地推理环境缺失，参见docs/medqa-local-qwen.md\n' >&2;exit 2
fi
if [[ ${1:-serve} == prepare ]]; then
  exec "$QWEN_PYTHON" -B scripts/prepare_qwen32b_int8.py --output "$QWEN_MODEL_DIR"
fi
if [[ ${1:-serve} != serve ]]; then printf '用法: bash scripts/serve_qwen32b_gpu0.sh [prepare|serve]\n' >&2;exit 2;fi
if [[ ! -f "$QWEN_MODEL_DIR/moreagent-deployment.json" ]]; then
  printf '请先运行: bash scripts/serve_qwen32b_gpu0.sh prepare\n' >&2;exit 2
fi
"$QWEN_PYTHON" -B - "$QWEN_MODEL_DIR" "$QWEN_PORT" "$QWEN_MAX_SEQS" "$QWEN_BATCH_TOKENS" <<'PY'
import json,socket,sys
from pathlib import Path
model=Path(sys.argv[1]);port=int(sys.argv[2])
if not 1 <= int(sys.argv[3]) <= 4 or int(sys.argv[4]) < 8192:
    raise SystemExit('并发数须为1至4，批处理token预算至少8192；需要更多并发时先独立验证显存')
config=json.loads((model/'config.json').read_text())
if not config.get('quantization_config',{}).get('load_in_8bit'):
    raise SystemExit('拒绝启动：目标不是已准备的INT8模型，不能在48GB卡上回退加载BF16')
with socket.socket() as sock:
    sock.bind(('127.0.0.1',port))
PY
exec "$QWEN_PYTHON" -B -m vllm.entrypoints.openai.api_server \
  --model "$QWEN_MODEL_DIR" --served-model-name qwen2.5-32b-instruct-int8 \
  --host 127.0.0.1 --port "$QWEN_PORT" --api-key local-moreagent \
  --dtype bfloat16 --quantization bitsandbytes --load-format bitsandbytes \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.88 --max-model-len 8192 \
  --max-num-seqs "$QWEN_MAX_SEQS" --max-num-batched-tokens "$QWEN_BATCH_TOKENS" \
  --enable-prefix-caching --enforce-eager --generation-config vllm \
  --guided-decoding-backend xgrammar --disable-log-requests
