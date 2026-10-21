#!/usr/bin/env bash
# 使用已有环境；所有动态库/离线设置只作用于当前子进程。
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_PATH=${STRUCKFLOW_PYTHON:-$PROJECT_ROOT/../miniconda3/envs/struckflow/bin/python}
if [[ ! -x "$PYTHON_PATH" ]]; then
  printf '未找到 struckflow，请设置 STRUCKFLOW_PYTHON 为已有解释器。\n' >&2
  exit 2
fi
ENV_PREFIX=$("$PYTHON_PATH" -B -c 'import sys; print(sys.prefix)')
cd -- "$PROJECT_ROOT"
if [[ ${1:-} == parallel ]]; then
  shift
  exec env LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PYTHON_PATH" -B scripts/run_medqa_parallel.py "$@"
fi
if [[ ${1:-} == test ]]; then
  shift
  exec env LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PYTHON_PATH" -B -m unittest LangChain.test.test_medqa LangChain.test.test_medqa_audit LangChain.test.test_medqa_experiment LangChain.test.test_medqa_answer_policy LangChain.test.test_medqa_checkpoint LangChain.test.test_medqa_v6 LangChain.test.test_medqa_results LangChain.test.test_medqa_bounded_path LangChain.test.test_medqa_handoff LangChain.test.test_medqa_memory_units LangChain.test.test_medqa_memory_selection LangChain.test.test_medqa_local LangChain.test.test_medqa_parallel LangChain.test.test_medqa_communication LangChain.test.test_medqa_workspace LangChain.test.test_medqa_need_first LangChain.test.test_medqa_dataset50 -v "$@"
fi
exec env LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON_PATH" -B -m LangChain.medqa.cli "$@"
