"""
eval/evaluator.py  — Evaluation framework for the clinical annotation pipeline.

Provides:
  - Gold-label comparison (accuracy, token-F1, Jaccard) per field
  - Inter-agent agreement scoring
  - Pipeline metadata analysis (debate rounds, vote agreement, sanity rate)
  - Ablation study helper (disable debate / consensus voters / review)
"""
from typing import Dict, Any, List, Optional


def accuracy(predictions: List[str], golds: List[str]) -> float:
    if not golds:
        return 0.0
    correct = sum(p.strip().lower() == g.strip().lower() for p, g in zip(predictions, golds))
    return correct / len(golds)


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    tp = len(pred_tokens & gold_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def list_jaccard(pred: List[str], gold: List[str]) -> float:
    p_set = {s.lower().strip() for s in pred}
    g_set = {s.lower().strip() for s in gold}
    if not p_set and not g_set:
        return 1.0
    if not p_set or not g_set:
        return 0.0
    return len(p_set & g_set) / len(p_set | g_set)


def evaluate_against_gold(
    predictions: List[Dict[str, Any]],
    gold_labels: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Full metric suite comparing predictions to gold annotations."""
    assert len(predictions) == len(gold_labels)

    def mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    stage_acc = accuracy(
        [p.get("stage_of_change", "") for p in predictions],
        [g.get("stage_of_change", "") for g in gold_labels],
    )
    concern_f1 = mean([token_f1(p.get("main_concern", ""), g.get("main_concern", "")) for p, g in zip(predictions, gold_labels)])
    trigger_f1 = mean([token_f1(p.get("primary_trigger", ""), g.get("primary_trigger", "")) for p, g in zip(predictions, gold_labels)])
    intervention_j = mean([list_jaccard(p.get("recommended_intervention", []), g.get("recommended_intervention", [])) for p, g in zip(predictions, gold_labels)])

    sanity_pass_rate = sum(1 for p in predictions if p.get("sanity_passed", True)) / len(predictions)
    confidences = [p["confidence_score"] for p in predictions if p.get("confidence_score") is not None]
    agreements = [p["vote_agreement"] for p in predictions if p.get("vote_agreement") is not None]
    debate_rounds = [p.get("debate_rounds_used", 0) for p in predictions]

    return {
        "n_samples": len(predictions),
        "stage_of_change_accuracy": round(stage_acc, 4),
        "main_concern_token_f1": concern_f1,
        "primary_trigger_token_f1": trigger_f1,
        "intervention_jaccard": intervention_j,
        "sanity_pass_rate": round(sanity_pass_rate, 4),
        "mean_confidence_score": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "mean_vote_agreement": round(sum(agreements) / len(agreements), 4) if agreements else None,
        "mean_debate_rounds": round(sum(debate_rounds) / len(debate_rounds), 2),
        "aggregate_score": mean([stage_acc, concern_f1, trigger_f1, intervention_j]),
    }


def run_ablation(
    conversations: List[str],
    ablate_node: str,
    gold_labels: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Disable a pipeline component and measure performance drop.

    ablate_node options:
      'debate'          -> set CONFIDENCE_DEBATE_THRESHOLD = 1.1 (never triggers)
      'multi_consensus' -> set CONSENSUS_VOTERS = 1
    """
    import annotate.config.settings as cfg

    originals = {}
    if ablate_node == "debate":
        originals["CONFIDENCE_DEBATE_THRESHOLD"] = cfg.CONFIDENCE_DEBATE_THRESHOLD
        cfg.CONFIDENCE_DEBATE_THRESHOLD = 1.1
    elif ablate_node == "multi_consensus":
        originals["CONSENSUS_VOTERS"] = cfg.CONSENSUS_VOTERS
        cfg.CONSENSUS_VOTERS = 1

    from annotate.graph.runner import run_pipeline
    predictions = []
    for i, conv in enumerate(conversations):
        try:
            predictions.append(run_pipeline(conv, conversation_id=f"ablation_{i:04d}"))
        except Exception as exc:
            predictions.append({"_error": str(exc)})

    # Restore
    for k, v in originals.items():
        setattr(cfg, k, v)

    result: Dict[str, Any] = {"ablate_node": ablate_node, "predictions": predictions}
    if gold_labels:
        valid = [(p, g) for p, g in zip(predictions, gold_labels) if "_error" not in p]
        if valid:
            vp, vg = zip(*valid)
            result["metrics"] = evaluate_against_gold(list(vp), list(vg))
    return result


def inter_agent_agreement(outputs_a: List[Dict], outputs_b: List[Dict], field: str = "stage_of_change") -> float:
    if not outputs_a or not outputs_b:
        return 0.0
    matches = sum(a.get(field, "").lower() == b.get(field, "").lower() for a, b in zip(outputs_a, outputs_b))
    return matches / min(len(outputs_a), len(outputs_b))
