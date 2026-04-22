"""
examples/generate_dataset.py
------------------------------
High-throughput dataset generator with concurrent processing,
per-item error recovery, and JSONL streaming output.

Usage:
    python -m annotate.examples.generate_dataset
    python -m annotate.examples.generate_dataset data/raw/corpus.json data/processed/training.jsonl 8
"""
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any


from annotate.graph.runner import run_pipeline
from annotate.utils.file_utils import load_json
from annotate.utils.logger import log, error

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def _process_item(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    conv_id = item.get("id", f"conv_{index:06d}")
    start = time.perf_counter()
    try:
        row = run_pipeline(item["conversation"], conversation_id=conv_id)
        row["_id"] = conv_id
        row["_status"] = "ok"
        row["_elapsed"] = round(time.perf_counter() - start, 2)
    except Exception as exc:
        row = {
            "_id": conv_id,
            "_status": "failed",
            "_error": str(exc),
            "_elapsed": round(time.perf_counter() - start, 2),
        }
        error(f"FAILED {conv_id}: {exc}")
    return row


def generate_dataset(input_path: str, output_path: str, workers: int = 4) -> None:
    data: List[Dict[str, Any]] = load_json(input_path)
    total = len(data)
    log(f"Starting dataset generation: {total} conversations, {workers} workers")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    succeeded, failed = 0, 0
    iterator = list(enumerate(data, 1))
    if HAS_TQDM:
        iterator = tqdm(iterator, desc="Annotating", unit="conv")

    with open(out, "w", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_item, item, i): i for i, item in enumerate(data, 1)}
            for future in as_completed(futures):
                row = future.result()
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                f_out.flush()
                if row.get("_status") == "ok":
                    succeeded += 1
                else:
                    failed += 1

    print(f"\n{'='*50}")
    print(f"Dataset generation complete")
    print(f"  Total:     {total}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {failed}")
    print(f"  Output:    {out.resolve()}")
    print(f"{'='*50}")


def main() -> None:
    input_path  = sys.argv[1] if len(sys.argv) > 1 else "data/raw/conversations.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/training_dataset.jsonl"
    workers     = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    generate_dataset(input_path, output_path, workers)


if __name__ == "__main__":
    main()
