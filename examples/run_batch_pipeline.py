"""
examples/run_batch_pipeline.py
--------------------------------
Process a JSON file of conversations and save annotated results.
Now uses a shared AgentMemory so per-agent stats accumulate across runs.

Usage:
    python -m examples.run_batch_pipeline
    python -m examples.run_batch_pipeline data/raw/conversations.json data/processed/output.json
"""
import sys
import json
import time
from pathlib import Path


from annotate.graph.runner import run_pipeline
from annotate.agents.memory.agent_memory import AgentMemory
from annotate.utils.file_utils import load_json, save_json, append_jsonl
from annotate.utils.logger import log, error
from annotate.config.settings import DATA_PATH, OUTPUT_PATH, FAILED_PATH


def main():
    input_path  = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    output_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PATH

    data = load_json(input_path)
    total = len(data)
    log(f"Loaded {total} conversation(s) from {input_path}")

    # Shared memory accumulates agent stats across the whole batch
    memory = AgentMemory(persist_path="data/processed/agent_memory.json")

    rows, failed = [], []

    for i, item in enumerate(data, 1):
        conv_id = item.get("id", f"conv_{i:04d}")
        log(f"[{i}/{total}] Processing {conv_id} ...")
        start = time.perf_counter()
        try:
            row = run_pipeline(item["conversation"], conversation_id=conv_id, memory=memory)
            row["_id"] = conv_id
            rows.append(row)
            append_jsonl(row, output_path.replace(".json", ".jsonl"))
            log(f"[{i}/{total}] done in {time.perf_counter() - start:.1f}s "
                f"| sanity={'OK' if row.get('sanity_passed') else 'FAIL'} "
                f"| debate_rounds={row.get('debate_rounds_used', 0)} "
                f"| vote_agreement={row.get('vote_agreement')}")
        except Exception as exc:
            elapsed = time.perf_counter() - start
            error(f"[{i}/{total}] FAILED {conv_id} after {elapsed:.1f}s: {exc}")
            failed.append({"id": conv_id, "error": str(exc)})

    save_json(rows, output_path)
    log(f"Saved {len(rows)} rows to {output_path}")

    if failed:
        save_json(failed, FAILED_PATH)
        error(f"{len(failed)} conversation(s) failed — see {FAILED_PATH}")

    # Print aggregate stats
    print(f"\n{'='*50}")
    print(f"Batch complete: {len(rows)}/{total} succeeded")
    if rows:
        sanity_ok   = sum(1 for r in rows if r.get("sanity_passed", True))
        mean_conf   = sum(r.get("confidence_score") or 0 for r in rows) / len(rows)
        mean_debate = sum(r.get("debate_rounds_used") or 0 for r in rows) / len(rows)
        mean_agree  = [r["vote_agreement"] for r in rows if r.get("vote_agreement") is not None]
        print(f"  Sanity passed:      {sanity_ok}/{len(rows)}")
        print(f"  Mean confidence:    {mean_conf:.3f}")
        print(f"  Mean debate rounds: {mean_debate:.2f}")
        if mean_agree:
            print(f"  Mean vote agreement:{sum(mean_agree)/len(mean_agree):.3f}")

    agent_stats = memory.agent_stats()
    if agent_stats:
        print(f"\nAgent memory stats:")
        for agent, s in agent_stats.items():
            print(f"  {agent}: runs={s['runs']}, mean_conf={s['mean_confidence']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
