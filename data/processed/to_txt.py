"""
data/processed/to_txt.py
--------------------------
Convert a JSONL output file (from run_batch_pipeline or generate_dataset)
to a human-readable .txt file.

Now includes the research metadata fields added in the pipeline upgrade.

Usage:
    python data/processed/to_txt.py output_gemma.jsonl output.txt
"""
import json
import sys
from pathlib import Path


def jsonl_to_txt(input_file: str, output_file: str) -> None:
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        sys.exit(1)

    count = 0
    with open(input_path, encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            count += 1

            outfile.write(f"Conversation ID: {data.get('_id', 'N/A')}\n")
            outfile.write(f"Stage of Change:        {data.get('stage_of_change', 'N/A')}\n")
            outfile.write(f"Concerns:               {', '.join(data.get('all_concerns', [])) or 'N/A'}\n")
            outfile.write(f"Triggers:               {', '.join(data.get('all_triggers', [])) or 'N/A'}\n")
            outfile.write(f"Quit Attempt History:   {data.get('quit_attempt_history', 'N/A')}\n")

            outfile.write("Recommended Interventions:\n")
            for i, item in enumerate(data.get("recommended_intervention", []), 1):
                outfile.write(f"  {i}. {item}\n")

            outfile.write(f"Reasoning Summary:      {data.get('reasoning_summary', 'N/A')}\n")
            outfile.write(f"Confidence Score:       {data.get('confidence_score', 'N/A')}\n")

            # Research metadata fields
            outfile.write("--- Pipeline Metadata ---\n")
            outfile.write(f"  Debate Rounds Used:   {data.get('debate_rounds_used', 'N/A')}\n")
            outfile.write(f"  Vote Agreement:       {data.get('vote_agreement', 'N/A')}\n")
            outfile.write(f"  Sanity Check Passed:  {data.get('sanity_passed', 'N/A')}\n")
            inconsistencies = data.get("inconsistencies", [])
            if inconsistencies:
                outfile.write("  Inconsistencies:\n")
                for issue in inconsistencies:
                    outfile.write(f"    - {issue}\n")
            else:
                outfile.write("  Inconsistencies:      None\n")

            if data.get("_status") == "failed":
                outfile.write(f"  [FAILED] Error: {data.get('_error', 'unknown')}\n")

            outfile.write("\n" + "=" * 80 + "\n\n")

    print(f"Conversion complete: {count} record(s) written to {output_path}")


if __name__ == "__main__":
    input_file  = sys.argv[1] if len(sys.argv) > 1 else "output_gemma.jsonl"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output.txt"
    jsonl_to_txt(input_file, output_file)
