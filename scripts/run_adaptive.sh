#!/bin/bash
# Adaptive boosting: global-adaptive (--use_global true) scales a single delta by
# context-query distribution difference; token-wise adaptive (--use_global false,
# the default) additionally weighs per-token attention (lambda1) and semantic
# similarity (lambda2 = 1 - lambda1).
# Usage: scripts/run_adaptive.sh --dataset {cnndm,xsum,nqswap,nqsynth} [--use_global true|false] [--min_delta N] [--max_delta N] [--lambda1 N] [--model_path PATH_OR_HF_ID] [--run_name NAME]
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    echo "Usage: $0 --dataset {cnndm,xsum,nqswap,nqsynth} [--use_global true|false] [--min_delta N] [--max_delta N] [--lambda1 N] [--model_path PATH_OR_HF_ID] [--run_name NAME]" >&2
    exit 1
}

DATASET=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --use_global) USE_GLOBAL="$2"; shift 2 ;;
        --min_delta) MIN_DELTA="$2"; shift 2 ;;
        --max_delta) MAX_DELTA="$2"; shift 2 ;;
        --lambda1) LAMBDA1="$2"; shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --run_name) RUN_NAME="$2"; shift 2 ;;
        *) usage ;;
    esac
done
[ -z "${DATASET}" ] && usage

# Decoding parameters differ between summarization datasets (cnndm, xsum)
# and QA datasets (nqswap, nqsynth); everything else is shared.
case "${DATASET}" in
    cnndm|xsum)   MAXCTXLEN="1948"; GENLEN="100"; TOPP="0.9"; EVAL_TASK="summarization" ;;
    nqswap|nqsynth) MAXCTXLEN="2038"; GENLEN="10";  TOPP="0.0"; EVAL_TASK="qa" ;;
    *) echo "Unknown dataset: ${DATASET}" >&2; usage ;;
esac
GLOBALLEN="2048"

FN_PREFIX="${FN_PREFIX:-${REPO_ROOT}/eval/${DATASET}_example_input/${DATASET}}"

hf_cache="${REPO_ROOT}/.cache/huggingface"
mkdir -p "${hf_cache}"
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_NAME="${RUN_NAME:-default}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/${RUN_NAME}}"
mkdir -p "${RESULTS_DIR}" "${OUTPUT_DIR}"

MIN_DELTA="${MIN_DELTA:-5.0}"
MAX_DELTA="${MAX_DELTA:-10.0}"
LAMBDA1="${LAMBDA1:-0.6}"
LAMBDA2=$(echo "1 - ${LAMBDA1}" | bc)
USE_GLOBAL="${USE_GLOBAL:-false}"

# Base model: defaults to Meta-Llama-3-8B-Instruct; pass --model_path to use
# another local checkpoint or HF model id.
MODEL_ARGS=()
if [ -n "${MODEL_PATH:-}" ]; then
    MODEL_ARGS+=(--force_model_name_or_path "${MODEL_PATH}")
fi

# Token-wise adaptive uses the occurrence-aggregation implementation described
# in the paper: attention summed over each token's occurrences, mean-normalized
# importance, renormalized component weights, and boosting restricted to the
# source span. These flags are no-ops when --use_global true.
TOKENWISE_ARGS=()
if [ "${USE_GLOBAL}" = "false" ]; then
    TOKENWISE_ARGS+=(
        --tokenwise_implementation occurrence_agg
        --token_attention_aggregation sum
        --token_importance_normalization mean1
        --renormalize_component_weights
        --boost_source_scope source_only
    )
fi

echo "Processing: dataset=${DATASET}, min_delta=${MIN_DELTA}, max_delta=${MAX_DELTA}, lambda1=${LAMBDA1}, lambda2=${LAMBDA2}, global=${USE_GLOBAL}"

WEIGHT="1_0"
TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
# Decode into a fresh subdirectory so the produced file can be located reliably
# (the decoder appends implementation-specific suffixes to the filename).
DECODE_DIR="${OUTPUT_DIR}/decode_$$"
mkdir -p "${DECODE_DIR}"

echo "Running decode..."
python "${REPO_ROOT}/src/group_decode_adaptive_fileio.py" \
    --max_seq_length ${GLOBALLEN} \
    --model_name_or_path dummy \
    --seed 2023 \
    --use_slow_tokenizer \
    --file_mode ${TESTFILE} \
    --decode_truncate_len ${MAXCTXLEN} \
    --decode_depth ${GENLEN} \
    --train_mode decode \
    --projection_top_p ${TOPP} \
    --min_delta ${MIN_DELTA} \
    --max_delta ${MAX_DELTA} \
    --lambda1 ${LAMBDA1} \
    --lambda2 ${LAMBDA2} \
    --use_global ${USE_GLOBAL} \
    --output_dir "${DECODE_DIR}" \
    "${MODEL_ARGS[@]}" \
    "${TOKENWISE_ARGS[@]}"

if [ $? -ne 0 ]; then
    echo "Error: Decode failed"
    exit 1
fi

OUTPUT_FILE="$(find "${DECODE_DIR}" -maxdepth 1 -type f -name '*.jsonl' | head -n 1)"
if [ -z "${OUTPUT_FILE}" ]; then
    echo "Error: no decode output found in ${DECODE_DIR}"
    exit 1
fi
mv "${OUTPUT_FILE}" "${OUTPUT_DIR}/"
OUTPUT_FILE="${OUTPUT_DIR}/$(basename "${OUTPUT_FILE}")"
rmdir "${DECODE_DIR}" 2>/dev/null
echo "Decode completed successfully: ${OUTPUT_FILE}"

RESULT_FILE="${RESULTS_DIR}/evaluate_results_delta${MIN_DELTA}-${MAX_DELTA}_l1${LAMBDA1}_l2${LAMBDA2}_global${USE_GLOBAL}.log"
echo "Running evaluate..."
python "${REPO_ROOT}/eval/evaluate.py" \
    --pred_path "${OUTPUT_FILE}" \
    --data_path "${FN_PREFIX}_${WEIGHT}.jsonl" \
    --task "${EVAL_TASK}" \
    2>&1 | tee "${RESULT_FILE}"

if [ $? -ne 0 ]; then
    echo "Error: Evaluate failed"
    exit 1
fi

echo "Results for min_delta=${MIN_DELTA}, max_delta=${MAX_DELTA}, lambda1=${LAMBDA1}, lambda2=${LAMBDA2}, global=${USE_GLOBAL}:"
cat "${RESULT_FILE}"
