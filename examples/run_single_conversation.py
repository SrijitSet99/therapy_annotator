"""
examples/run_single_conversation.py
------------------------------------
Run the full adaptive pipeline on a single conversation and print all results,
including the new research metadata fields introduced in the upgrade.

Usage:
    python -m examples.run_single_conversation
    python -m examples.run_single_conversation path/to/chat.txt
"""
import json
import sys
from pathlib import Path


from annotate.graph.runner import run_pipeline
from annotate.agents.memory.agent_memory import AgentMemory


def _fmt(val, indent=2):
    return json.dumps(val, indent=indent, ensure_ascii=False)


def main():
    chat_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_chat.txt"
    print(f"Loading conversation from: {chat_path}\n")
    chat = Path(chat_path).read_text(encoding="utf-8")

    # Shared memory (persists across calls in this session)
    memory = AgentMemory()

    print("Running adaptive pipeline ...\n")
    result = run_pipeline(chat, conversation_id="single_run", memory=memory)

    # ── Core clinical fields ──────────────────────────────────────────────
    print("=" * 60)
    print("CLINICAL ANNOTATION")
    print("=" * 60)
    clinical_fields = [
        "stage_of_change", "all_concerns", "all_triggers",
        "quit_attempt_history", "recommended_intervention",
        "reasoning_summary", "confidence_score",
    ]
    for field in clinical_fields:
        val = result.get(field)
        if isinstance(val, list):
            print(f"\n{field}:")
            for i, item in enumerate(val, 1):
                print(f"  {i}. {item}")
        else:
            print(f"\n{field}: {val}")

    # ── Research metadata ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE METADATA (research fields)")
    print("=" * 60)
    meta_fields = [
        "debate_rounds_used", "vote_agreement",
        "sanity_passed", "inconsistencies",
    ]
    for field in meta_fields:
        val = result.get(field)
        print(f"{field}: {val}")

    # ── Agent memory stats ────────────────────────────────────────────────
    stats = memory.agent_stats()
    if stats:
        print("\n" + "=" * 60)
        print("AGENT MEMORY STATS")
        print("=" * 60)
        for agent, s in stats.items():
            print(f"  {agent}: runs={s['runs']}, mean_conf={s['mean_confidence']}, fail_rate={s['failure_rate']}")

    print("\n" + "=" * 60)
    print("FULL JSON OUTPUT")
    print("=" * 60)
    print(_fmt(result))


if __name__ == "__main__":
    main()
