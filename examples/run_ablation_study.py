"""
examples/run_ablation_study.py
--------------------------------
Runs a structured ablation study comparing:
  1. Full pipeline (debate + multi-consensus + sanity check)
  2. No debate   (confidence threshold set to 1.1 — never triggers)
  3. Single voter (CONSENSUS_VOTERS = 1)

For each condition, prints per-field metrics side by side.
If a gold JSONL path is provided, computes F1/accuracy against it.

Usage (no gold labels — shows pipeline metadata comparison only):
    python -m examples.run_ablation_study

Usage (with gold labels):
    python -m examples.run_ablation_study data/eval/gold.json
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any


from annotate.utils.file_utils import load_json
from annotate.utils.logger import log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _pipeline_metadata_summary(rows: List[Dict[str, Any]], label: str) -> None:
    n = len(rows)
    ok = [r for r in rows if "_error" not in r]
    debate_rounds = [r.get("debate_rounds_used", 0) or 0 for r in ok]
    agreements    = [r["vote_agreement"] for r in ok if r.get("vote_agreement") is not None]
    confs         = [r["confidence_score"] for r in ok if r.get("confidence_score") is not None]
    sanity_ok     = sum(1 for r in ok if r.get("sanity_passed", True))

    print(f"\n  [{label}]")
    print(f"    Succeeded:          {len(ok)}/{n}")
    print(f"    Mean debate rounds: {_mean(debate_rounds)}")
    print(f"    Mean vote agreement:{_mean(agreements)}")
    print(f"    Mean confidence:    {_mean(confs)}")
    print(f"    Sanity pass rate:   {sanity_ok}/{len(ok)}")


def _run_condition(conversations: List[str], label: str, patches: Dict[str, Any]) -> List[Dict]:
    """Apply config patches, run pipeline on all conversations, restore patches."""
    import annotate.config.settings as cfg

    originals = {}
    for k, v in patches.items():
        originals[k] = getattr(cfg, k)
        setattr(cfg, k, v)
        log(f"ablation: patched {k}={v}")

    # Force graph rebuild with new config
    import annotate.graph.runner as runner_mod
    runner_mod._graph = None

    from annotate.graph.runner import run_pipeline
    results = []
    for i, conv in enumerate(conversations):
        try:
            row = run_pipeline(conv, conversation_id=f"abl_{label}_{i:04d}")
            results.append(row)
        except Exception as exc:
            results.append({"_error": str(exc)})

    # Restore config and graph
    for k, v in originals.items():
        setattr(cfg, k, v)
    runner_mod._graph = None

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    gold_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Load conversations
    data_path = "data/raw/conversations.json"
    data = load_json(data_path)
    conversations = [item["conversation"] for item in data]

    if not conversations:
        print("No conversations found in data/raw/conversations.json")
        sys.exit(1)

    gold_labels = None
    if gold_path:
        gold_labels = json.loads(Path(gold_path).read_text())
        assert len(gold_labels) == len(conversations), \
            f"Gold ({len(gold_labels)}) must match conversations ({len(conversations)})"

    print("=" * 60)
    print("ABLATION STUDY")
    print(f"  Conversations: {len(conversations)}")
    print("=" * 60)

    conditions = [
        ("full_pipeline",   {}),
        ("no_debate",       {"CONFIDENCE_DEBATE_THRESHOLD": 1.1}),
        ("single_voter",    {"CONSENSUS_VOTERS": 1}),
        ("no_debate_single",{"CONFIDENCE_DEBATE_THRESHOLD": 1.1, "CONSENSUS_VOTERS": 1}),
    ]

    all_results: Dict[str, List[Dict]] = {}

    for label, patches in conditions:
        print(f"\nRunning condition: {label} ...")
        rows = _run_condition(conversations, label, patches)
        all_results[label] = rows
        _pipeline_metadata_summary(rows, label)

    # ── Metric comparison (if gold provided) ─────────────────────────────
    if gold_labels:
        from annotate.eval.evaluator import evaluate_against_gold
        print("\n" + "=" * 60)
        print("METRIC COMPARISON vs GOLD")
        print("=" * 60)
        metric_keys = [
            "stage_of_change_accuracy",
            "all_concerns_jaccard",
            "all_triggers_jaccard",
            "intervention_jaccard",
            "aggregate_score",
        ]
        # Header
        header = f"{'Metric':<35}" + "".join(f"{c:<20}" for c, _ in conditions)
        print(header)
        print("-" * (35 + 20 * len(conditions)))

        per_condition_metrics = {}
        for label, _ in conditions:
            valid = [r for r in all_results[label] if "_error" not in r]
            if valid:
                per_condition_metrics[label] = evaluate_against_gold(valid, gold_labels[:len(valid)])
            else:
                per_condition_metrics[label] = {}

        for key in metric_keys:
            row_str = f"{key:<35}"
            for label, _ in conditions:
                val = per_condition_metrics.get(label, {}).get(key, "N/A")
                row_str += f"{str(val):<20}"
            print(row_str)

    print("\n" + "=" * 60)
    print("Ablation study complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
