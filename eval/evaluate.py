import argparse
from collections import defaultdict
import json
import logging
import re
import statistics
import string
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rouge_score import rouge_scorer
from bert_score import BERTScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FACTKB_MODEL = "bunsenfeng/FactKB"
FACTKB_TOKENIZER = "roberta-base"
BERTSCORE_MODEL = "roberta-base"
BERTSCORE_NUM_LAYERS = 9


def load_gold_map(dataset_path):
    """Map input_index -> gold example (the with-context line of each pair)."""
    gold = {}
    with open(dataset_path) as f:
        for line in f:
            ex = json.loads(line)
            if int(ex.get("assigned_process", -1)) != 0:
                continue
            gold[int(ex["input_index"])] = ex
    return gold


def load_prediction_map(pred_path):
    pred = {}
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            value = obj.get("string", "")
            if isinstance(value, list):
                value = value[0] if value else ""
            pred[int(obj["input_index"])] = str(value)
    return pred


def get_article_text(ex):
    article = ex.get("article")
    if isinstance(article, str) and article.strip():
        return article.strip()
    return str(ex.get("context_string", "")).strip()


def get_gold_answers(ex):
    gold = ex.get("gold_answers", "")
    if isinstance(gold, list):
        return [str(item) for item in gold]
    if gold is None:
        return [""]
    return [str(gold)]


def normalize_answer(text):
    text = str(text).split("\n\n")[0]
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def calculate_acc(prediction, ground_truths):
    norm_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) in norm_pred:
            return 1
    return 0


def compute_factkb_scores(tokenizer, model, device, predictions, documents, batch_size=64):
    scores = []
    with torch.no_grad():
        for start in range(0, len(predictions), batch_size):
            batch_preds = predictions[start:start + batch_size]
            batch_docs = documents[start:start + batch_size]
            enc = tokenizer(
                batch_preds,
                batch_docs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=1)[:, 1]
            scores.extend(probs.detach().cpu().tolist())
    return scores


def compute_bert_p_scores(scorer, predictions, documents, batch_size=32):
    """BERTScore precision of each prediction against its source document."""
    scores = [0.0] * len(predictions)
    valid = [i for i, p in enumerate(predictions) if str(p).strip()]
    if not valid:
        return scores
    precision, _, _ = scorer.score(
        [predictions[i] for i in valid],
        [documents[i] for i in valid],
        verbose=False,
        batch_size=batch_size,
    )
    for i, s in zip(valid, precision.detach().cpu().tolist()):
        scores[i] = s
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the gold input jsonl file.")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to the model's prediction output jsonl file.")
    parser.add_argument(
        "--task",
        choices=["summarization", "qa"],
        default="summarization",
        help="qa additionally reports answer accuracy (normalized substring match).",
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    gold_map = load_gold_map(args.data_path)
    pred_map = load_prediction_map(args.pred_path)
    indices = sorted(set(gold_map) & set(pred_map))
    if len(indices) < len(gold_map):
        logger.warning("Only %d/%d gold examples have predictions.", len(indices), len(gold_map))

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_values = defaultdict(list)
    predictions, documents, acc_scores = [], [], []
    for idx in indices:
        pred = pred_map[idx]
        gold_ex = gold_map[idx]
        answers = get_gold_answers(gold_ex)
        score = rouge.score(pred, answers[0] if answers else "")
        for key, value in score.items():
            rouge_values[key].append(value.fmeasure)
        predictions.append(pred)
        documents.append(get_article_text(gold_ex))
        if args.task == "qa":
            acc_scores.append(calculate_acc(pred, answers))

    device = torch.device(args.device)
    logger.info("Loading FactKB (%s)...", FACTKB_MODEL)
    factkb_tokenizer = AutoTokenizer.from_pretrained(FACTKB_TOKENIZER)
    factkb = AutoModelForSequenceClassification.from_pretrained(FACTKB_MODEL, num_labels=2).to(device)
    factkb.eval()
    factkb_scores = compute_factkb_scores(factkb_tokenizer, factkb, device, predictions, documents)

    logger.info("Loading BERTScore (%s, layer %d)...", BERTSCORE_MODEL, BERTSCORE_NUM_LAYERS)
    bert_scorer = BERTScorer(
        model_type=BERTSCORE_MODEL,
        num_layers=BERTSCORE_NUM_LAYERS,
        device=args.device,
        lang="en",
        rescale_with_baseline=False,
    )
    bert_p_scores = compute_bert_p_scores(bert_scorer, predictions, documents)

    results = {
        "count": len(indices),
        "rouge1": statistics.mean(rouge_values["rouge1"]),
        "rouge2": statistics.mean(rouge_values["rouge2"]),
        "rougeL": statistics.mean(rouge_values["rougeL"]),
        "factkb": statistics.mean(factkb_scores),
        "bert_p": statistics.mean(bert_p_scores),
    }
    if acc_scores:
        results["qa_acc"] = statistics.mean(acc_scores)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
