"""
examples/run_multisession.py
------------------------------
Demonstrates the multi-session longitudinal pipeline.

Runs 4 counseling sessions for the same patient sequentially.
After each session prints:
  - Current annotation
  - Stage trajectory so far
  - What changed since last session (delta)
  - Any longitudinal flags raised

Usage:
    python -m examples.run_multisession
    python -m examples.run_multisession data/raw/multisession_sample.json patient_001
"""
import sys
import json
from pathlib import Path
from typing import Dict

from annotate.graph.runner import run_patient_sessions
from annotate.agents.memory.patient_memory import PatientMemory
from annotate.utils.file_utils import load_json


def _divider(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _print_session_result(result: Dict, session_number: int) -> None:
    row    = result.get("dataset_row", {})
    long_o = result.get("longitudinal_output", {})
    delta  = long_o.get("delta") or {}

    _divider(f"SESSION {session_number} — {result.get('patient_id', '?')}")

    # Core annotation
    print(f"\n  Stage of change:       {row.get('stage_of_change')}")
    print(f"  Main concern:          {row.get('main_concern')}")
    print(f"  Primary trigger:       {row.get('primary_trigger')}")
    print(f"  Reasoning:             {row.get('reasoning_summary')}")
    print(f"  Confidence:            {row.get('confidence_score')}")

    if row.get("recommended_intervention"):
        print(f"\n  Interventions:")
        for i, item in enumerate(row["recommended_intervention"], 1):
            print(f"    {i}. {item}")

    # Trajectory (sessions 2+)
    trajectory = long_o.get("stage_trajectory", [])
    if trajectory:
        print(f"\n  Stage trajectory:      {' → '.join(trajectory)}")
        print(f"  Overall progress:      {long_o.get('overall_progress', '?')}")

    # Delta (sessions 2+)
    if delta and delta.get("stage_before"):
        print(f"\n  ── Delta from session {delta.get('from_session')} ──")
        print(f"  Stage direction:       {delta.get('stage_direction')}")
        if delta.get("new_triggers"):
            print(f"  New triggers:          {delta['new_triggers']}")
        if delta.get("resolved_triggers"):
            print(f"  Resolved triggers:     {delta['resolved_triggers']}")
        if delta.get("new_concerns"):
            print(f"  New concerns:          {delta['new_concerns']}")
        if delta.get("resolved_concerns"):
            print(f"  Resolved concerns:     {delta['resolved_concerns']}")
        if delta.get("clinical_summary"):
            print(f"\n  Clinical summary:")
            print(f"    {delta['clinical_summary']}")

    # Flags
    flags = long_o.get("longitudinal_flags", [])
    if flags:
        print(f"\n  ⚠ Longitudinal flags:")
        for f in flags:
            print(f"    • {f}")
    else:
        print(f"\n  ✓ No longitudinal flags")

    # Pipeline metadata
    print(f"\n  Debate rounds used:    {row.get('debate_rounds_used', 0)}")
    print(f"  Vote agreement:        {row.get('vote_agreement')}")
    print(f"  Within-session sanity: {'PASS' if row.get('sanity_passed') else 'FAIL'}")
    print(f"  Cross-session sanity:  {'PASS' if long_o.get('longitudinal_flags') == [] else 'FLAGS'}")


# Allow Dict typing
from typing import Dict


def main():
    data_path  = sys.argv[1] if len(sys.argv) > 1 else "data/raw/multisession_sample.json"
    patient_id = sys.argv[2] if len(sys.argv) > 2 else "patient_demo_001"

    sessions = load_json(data_path)
    print(f"\nLoaded {len(sessions)} session(s) for patient '{patient_id}'")
    print(f"Data: {data_path}")

    pm = PatientMemory()

    # Check if patient has prior history
    profile = pm.load_or_create(patient_id)
    if profile.session_count > 0:
        print(f"\n[Resuming — patient has {profile.session_count} prior session(s)]")
        print(f"  Existing trajectory: {' → '.join(profile.stage_trajectory)}")

    # Run all sessions
    results = run_patient_sessions(sessions, patient_id=patient_id, patient_memory=pm)

    # Print results
    for i, result in enumerate(results, 1):
        _print_session_result(result, i)

    # Final profile summary
    profile = pm.load_or_create(patient_id)
    _divider("FINAL PATIENT PROFILE SUMMARY")
    print(f"\n  Patient ID:            {profile.patient_id}")
    print(f"  Total sessions:        {profile.total_sessions}")
    print(f"  Stage trajectory:      {' → '.join(profile.stage_trajectory)}")
    print(f"  Best stage reached:    {profile.best_stage_reached}")
    print(f"  Relapse count:         {profile.relapse_count}")
    print(f"  Stagnation count:      {profile.stagnation_count}")
    if profile.persistent_triggers:
        print(f"  Persistent triggers:   {', '.join(profile.persistent_triggers)}")
    if profile.persistent_concerns:
        print(f"  Persistent concerns:   {', '.join(profile.persistent_concerns)}")
    if profile.interventions_tried:
        print(f"  Interventions tried:   {', '.join(profile.interventions_tried[:5])}")
    print(f"\n  Profile saved to:      data/patients/{patient_id}.json")


if __name__ == "__main__":
    main()
