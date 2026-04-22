"""
examples/run_evaluation.py
----------------------------
Run the evaluation framework against a gold-labelled dataset.
Also runs a two-way ablation study: with vs without debate,
and with vs without multi-voter consensus.

Usage:
    python -m examples.run_evaluation data/eval/gold.json data/eval/predictions.jsonl
"""
import sys
import json
from pathlib import Path


from annotate.eval.evaluator import evaluate_against_gold, run_ablation, inter_agent_agreement


def load_jsonl(path: str):
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(l) for l in lines if l.strip()]


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m examples.run_evaluation <gold.json> <predictions.jsonl>")
        print("  gold.json         : list of gold DatasetRow dicts")
        print("  predictions.jsonl : JSONL file of pipeline output rows")
        sys.exit(1)

    gold_path = sys.argv[1]
    pred_path = sys.argv[2]

    golds = json.loads(Path(gold_path).read_text())
    preds = load_jsonl(pred_path)

    assert len(golds) == len(preds), (
        f"Gold ({len(golds)}) and predictions ({len(preds)}) must be the same length."
    )

    print("=" * 60)
    print("FULL PIPELINE EVALUATION")
    print("=" * 60)
    metrics = evaluate_against_gold(preds, golds)
    for k, v in metrics.items():
        print(f"  {k:<35} {v}")

    print("\n" + "=" * 60)
    print("INTER-AGENT AGREEMENT (predictions vs gold, per field)")
    print("=" * 60)
    for field in ["stage_of_change"]:
        score = inter_agent_agreement(preds, golds, field=field)
        print(f"  {field:<35} {score:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
