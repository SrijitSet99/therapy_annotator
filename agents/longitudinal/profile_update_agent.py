"""
agents/longitudinal/profile_update_agent.py
---------------------------------------------
Final node in the longitudinal pipeline. Runs after delta_agent and longitudinal_sanity_agent.

Responsibilities:
  1. Inject longitudinal metadata into the DatasetRow
  2. Build the LongitudinalAnnotation (full multi-session output)
  3. Create a SessionRecord from the completed DatasetRow
  4. Append the SessionRecord to the PatientProfile
  5. Persist the updated PatientProfile via PatientMemory

After this node, the state contains:
  dataset_row          — enriched with longitudinal metadata
  longitudinal_output  — full LongitudinalAnnotation dict
  patient_profile      — updated with the new session appended
"""
from __future__ import annotations

from typing import Dict, Any

from annotate.agents.memory.patient_memory import PatientMemory
from annotate.utils.logger import log, warning, log_data
from annotate.schemas.longitudinal_schema import LongitudinalAnnotation


def profile_update_agent(
    state: Dict[str, Any],
    patient_memory: PatientMemory,
) -> Dict[str, Any]:
    """
    Update the PatientProfile with the just-completed session and assemble
    the final LongitudinalAnnotation.
    """
    profile      = state.get("patient_profile")
    dataset_row  = state.get("dataset_row", {})
    delta        = state.get("session_delta")
    long_sanity  = state.get("longitudinal_sanity", {})
    conversation = state.get("chat", "")
    conv_id      = state.get("conversation_id")

    if profile is None:
        log("profile_update_agent: no patient_profile in state — skipping update.")
        return {}

    session_number = profile.session_count + 1
    log(f"profile_update_agent: finalising session {session_number} "
        f"for patient '{profile.patient_id}'")

    # ── 1. Enrich DatasetRow with longitudinal metadata ───────────────
    longitudinal_flags = state.get("longitudinal_flags", [])
    sanity_flags       = long_sanity.get("flags", [])
    all_flags          = longitudinal_flags + sanity_flags

    dataset_row["session_number"]             = session_number
    dataset_row["longitudinal_flags"]         = all_flags
    # FIX 1: was `long_sanity.get("passed", True)` — silently returns True when
    # the key is absent, masking a missing sanity check.  Default to False so
    # any absent result is treated as a failure rather than a silent pass.
    dataset_row["longitudinal_sanity_passed"] = long_sanity.get("passed", False)

    if delta:
        dataset_row["stage_direction"]          = delta.get("stage_direction", "unknown")
        dataset_row["new_triggers"]             = delta.get("new_triggers", [])
        dataset_row["resolved_triggers"]        = delta.get("resolved_triggers", [])
        dataset_row["new_concerns"]             = delta.get("new_concerns", [])
        dataset_row["resolved_concerns"]        = delta.get("resolved_concerns", [])
        dataset_row["new_interventions"]        = delta.get("new_interventions", [])
        dataset_row["continuing_interventions"] = delta.get("continuing_interventions", [])
        dataset_row["dropped_interventions"]    = delta.get("dropped_interventions", [])
        dataset_row["clinical_summary"]         = delta.get("clinical_summary", "")

    # ── 2. Build LongitudinalAnnotation ──────────────────────────────
    # FIX 2: _overall_progress must be called BEFORE append_session (step 4),
    # because it reads profile.stage_direction_history from the *pre-update*
    # profile.  After append_session the history already includes the current
    # session, so the delta direction would be double-counted.
    overall_progress = _overall_progress(profile, delta)

    # FIX 3: stage_trajectory was built by concatenating the existing list with
    # a one-element list — correct — but the existing list came from
    # profile.stage_trajectory which is the *pre-update* snapshot.  That is
    # intentional here (we haven't called append_session yet), so this is fine
    # as written.  However, the trajectory should use the resolved stage from
    # the delta (stage_after) when available, not the raw dataset_row value,
    # because delta_agent may have corrected an "unknown" stage.
    resolved_stage = (
        delta.get("stage_after", dataset_row.get("stage_of_change", "unknown"))
        if delta
        else dataset_row.get("stage_of_change", "unknown")
    )

    long_output = LongitudinalAnnotation(
        patient_id=profile.patient_id,
        session_number=session_number,
        current_annotation=dataset_row,
        delta=delta,
        stage_trajectory=profile.stage_trajectory + [resolved_stage],
        persistent_triggers=profile.persistent_triggers,
        persistent_concerns=profile.persistent_concerns,
        interventions_tried=profile.interventions_tried,
        relapse_detected=delta.get("relapse_detected", False) if delta else False,
        stagnation_detected=delta.get("stagnation_detected", False) if delta else False,
        overall_progress=overall_progress,
        longitudinal_flags=all_flags,
    )

    # ── 3. Build SessionRecord ────────────────────────────────────────
    record = PatientMemory.build_session_record(
        dataset_row=dataset_row,
        conversation=conversation,
        conversation_id=conv_id,
        all_triggers=state.get("triggers", {}).get("triggers", []),
        all_concerns=state.get("concern", {}).get("concerns", []),
    )

    # ── 4. Append to profile and recompute derived fields ─────────────
    profile = patient_memory.append_session(profile, record)

    # ── 5. Persist ───────────────────────────────────────────────────
    try:
        patient_memory.save(profile)
    except Exception as exc:
        warning(f"profile_update_agent: save failed — {exc}")

    log(f"profile_update_agent: done | trajectory={profile.stage_trajectory} | "
        f"stagnation={profile.stagnation_count} | overall={overall_progress}")

    payload = {
        "dataset_row":         dataset_row,
        "longitudinal_output": long_output.model_dump(),
        "patient_profile":     profile,
    }
    log_data(
        "profile_update_agent.output",
        {
            "dataset_row": dataset_row,
            "longitudinal_output": long_output.model_dump(),
            "patient_profile_id": getattr(profile, "patient_id", None),
            "session_count": getattr(profile, "session_count", None),
        },
    )
    return payload


def _overall_progress(profile, delta) -> str:
    """Classify overall patient trajectory for the LongitudinalAnnotation."""
    if not profile.stage_direction_history and delta is None:
        return "unknown"

    history = list(profile.stage_direction_history)
    if delta:
        history.append(delta.get("stage_direction", "stable"))

    if not history:
        return "unknown"

    # FIX 4: "relapse" is a stage name, not a direction label — it will never
    # appear in stage_direction_history (which holds values like "progress",
    # "regression", "stable", "recovery").  The relapse flag lives on the delta
    # and on profile.sessions[].  Check the delta and the last few session
    # records instead of scanning direction strings.
    recent_sessions = profile.sessions[-3:] if profile.sessions else []
    recent_relapse = any(
        getattr(s, "stage_of_change", "") == "relapse" for s in recent_sessions
    )
    if recent_relapse or (delta and delta.get("relapse_detected", False)):
        return "relapse"

    # Count progress vs regression in last 3 direction entries
    recent = history[-3:]
    progress_count   = recent.count("progress") + recent.count("recovery")
    regression_count = recent.count("regression")
    stable_count     = recent.count("stable")

    if progress_count > regression_count:
        return "improving"
    if regression_count > progress_count:
        return "regression"
    if stable_count == len(recent):
        return "stable"
    return "mixed"
