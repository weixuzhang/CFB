#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_HOME:-${HF_HUB_CACHE:-${REPO_ROOT}/.cache/huggingface}}"
mkdir -p "${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export CUDA_VISIBLE_DEVICES=0


## run evaluate
# Usage: PRED_PATH=<predictions.jsonl> GOLD_DATA_PATH=<gold.jsonl> [TASK=summarization|qa] bash scripts/run_evaluation.sh

if [ -z "${PRED_PATH}" ] || [ -z "${GOLD_DATA_PATH}" ]; then
  echo "Usage: PRED_PATH=<predictions.jsonl> GOLD_DATA_PATH=<gold.jsonl> [TASK=summarization|qa] bash scripts/run_evaluation.sh" >&2
  exit 1
fi

python "${REPO_ROOT}/eval/evaluate.py" --pred_path "${PRED_PATH}" --data_path "${GOLD_DATA_PATH}" --task "${TASK:-summarization}"


