#!/usr/bin/env bash
# Core regressions in the existing environment. No datasets or paid model calls.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
用法：bash scripts/test_memory.sh [smoke] [--python PATH]
运行通用记忆与工作流回归；默认寻找已有 Conda struckflow。
旧 retrieval/all 模式及自编任务数据已移除。
不安装依赖、不调用付费 API；临时数据库由测试自动清理。
EOF
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_PATH=${STRUCKFLOW_PYTHON:-}
if [[ ${1:-} == smoke ]]; then shift; fi
while (( $# )); do
  case "$1" in
    --python)
      if (( $# < 2 )); then printf '缺少 --python 参数\n' >&2; exit 2; fi
      PYTHON_PATH=$2
      shift 2
      ;;
    --help|-h) usage; exit 0 ;;
    *) printf '不支持的参数：%s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$PYTHON_PATH" ]]; then
  if [[ ${CONDA_DEFAULT_ENV:-} == struckflow && -x ${CONDA_PREFIX:-}/bin/python ]]; then
    PYTHON_PATH=$CONDA_PREFIX/bin/python
  elif [[ -x "$PROJECT_ROOT/../miniconda3/envs/struckflow/bin/python" ]]; then
    PYTHON_PATH=$PROJECT_ROOT/../miniconda3/envs/struckflow/bin/python
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
    PYTHON_PATH=$CONDA_BASE/envs/struckflow/bin/python
  fi
fi
if [[ -z "$PYTHON_PATH" || ! -x "$PYTHON_PATH" ]]; then
  printf '未找到已有 struckflow 环境，请用 --python 指定解释器。\n' >&2
  exit 2
fi
ENV_PREFIX=$("$PYTHON_PATH" -B -c 'import sys; print(sys.prefix)')
cd -- "$PROJECT_ROOT"
printf '运行通用回归检查；Python：%s\n' "$PYTHON_PATH"
exec env LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON_PATH" -B -m unittest \
    LangChain.test.test_memory_v2 \
    LangChain.test.test_memory_efficiency \
    LangChain.test.test_workflow_verification -v
