## Context Fidelity Boosting (CFB)

A method for improving context fidelity in LLM generation by boosting the logits of context tokens during decoding.

### Setup
```bash
pip install -r requirements.txt
```

### Boosting Modes
- **Static** (`scripts/run_static.sh`) — fixed boost delta for all context tokens.
- **Adaptive** (`scripts/run_adaptive.sh`) — delta scaled by context-query distribution difference (JSD, computed once per example from the with/without-context next-token distributions). `--use_global true` for a single global delta (boosts the full prompt); `--use_global false` (default) additionally weighs each token by attention (`lambda1`) and semantic similarity (`lambda2 = 1 - lambda1`), aggregated over token occurrences and restricted to the resolved source span — this is the configuration used for the paper's results.

Base model defaults to Meta-Llama-3-8B-Instruct; pass `--model_path <local dir or HF id>` to either script to use a different checkpoint (e.g. Mistral-7B-Instruct-v0.3, Llama-2-13b-chat-hf).

### Input format
One JSON object per line, with and without context, sharing an `input_index`:
```json
{"input_index": 0, "assigned_model": "huggyllama/llama-7b", "assigned_process": 0, "context_string": "<context> <question>", "assigned_weight": 2}
{"input_index": 0, "assigned_model": "huggyllama/llama-7b", "assigned_process": 1, "context_string": "<question>", "assigned_weight": -1}
```

### Data
Datasets aren't bundled. Bring your own file(s) matching the format, example datasets like CNN-DM, XSum, NQ-Swap, or NQ-Synth.

### Running experiments
```bash
scripts/run_static.sh --dataset cnndm --boost_delta 5.0
scripts/run_adaptive.sh --dataset nqswap --min_delta 1.0 --max_delta 10.0 --lambda1 0.6 --use_global false
```
Output goes to `output/${RUN_NAME}/`, results to `results/${RUN_NAME}/` (`RUN_NAME` defaults to `default`).

### Evaluation
Runs automatically after decoding, reporting ROUGE-1/2/L, FactKB, and BERTScore precision against the source document (`bert_p`); QA datasets additionally report answer accuracy (normalized substring match against any gold answer). To evaluate a prediction file standalone:
```bash
PRED_PATH=<pred.jsonl> GOLD_DATA_PATH=<gold.jsonl> bash scripts/run_evaluation.sh
```
