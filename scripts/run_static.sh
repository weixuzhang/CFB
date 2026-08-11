#!/bin/bash
# Static boosting: a fixed boost delta applied to all context tokens.
# Usage: scripts/run_static.sh --dataset {cnndm,xsum,nqswap,nqsynth} [--boost_delta N] [--model_path PATH_OR_HF_ID] [--run_name NAME]
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    echo "Usage: $0 --dataset {cnndm,xsum,nqswap,nqsynth} [--boost_delta N] [--model_path PATH_OR_HF_ID] [--run_name NAME]" >&2
    exit 1
}

DATASET=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --boost_delta) BOOST_DELTA="$2"; shift 2 ;;
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

# Fixed context boost delta
BOOST_DELTA="${BOOST_DELTA:-5.0}"

# Base model: defaults to Meta-Llama-3-8B-Instruct; pass --model_path to use
# another local checkpoint or HF model id.
MODEL_ARGS=()
if [ -n "${MODEL_PATH:-}" ]; then
    MODEL_ARGS+=(--force_model_name_or_path "${MODEL_PATH}")
fi

WEIGHT="1_0"
TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
# Decode into a fresh subdirectory so the produced file can be located reliably.
DECODE_DIR="${OUTPUT_DIR}/decode_$$"
mkdir -p "${DECODE_DIR}"

echo "Running decode (dataset=${DATASET}, boost_delta=${BOOST_DELTA})..."
python "${REPO_ROOT}/src/group_decode_static_fileio.py" \
    --max_seq_length ${GLOBALLEN} \
    --model_name_or_path dummy \
    --seed 2023 \
    --use_slow_tokenizer \
    --file_mode ${TESTFILE} \
    --decode_truncate_len ${MAXCTXLEN} \
    --decode_depth ${GENLEN} \
    --train_mode decode \
    --projection_top_p ${TOPP} \
    --context_boost_delta ${BOOST_DELTA} \
    --output_dir "${DECODE_DIR}" \
    "${MODEL_ARGS[@]}"

if [ $? -ne 0 ]; then
    echo "Error: Decode failed for boost delta ${BOOST_DELTA}"
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

RESULT_FILE="${RESULTS_DIR}/evaluate_results_boost${BOOST_DELTA}.log"
echo "Running evaluate..."
python "${REPO_ROOT}/eval/evaluate.py" \
    --pred_path "${OUTPUT_FILE}" \
    --data_path "${FN_PREFIX}_${WEIGHT}.jsonl" \
    --task "${EVAL_TASK}" \
    2>&1 | tee "${RESULT_FILE}"

if [ $? -ne 0 ]; then
    echo "Error: Evaluate failed for boost delta ${BOOST_DELTA}"
    exit 1
fi

echo "Results for boost delta ${BOOST_DELTA}:"
cat "${RESULT_FILE}"
