"""
agents/longitudinal/context_agent.py
--------------------------------------
Runs BEFORE the extractor fan-out on sessions 2+.

Takes the PatientProfile and enriches the GraphState with:
  1. A structured context_summary string (already built by PatientMemory)
  2. A session-aware version of the conversation text that extractors receive
  3. Flags for the manager about what to watch for (e.g. relapse risk)

For session 1 (no history) this agent is a no-op — it returns the state
unchanged so the graph can use a single workflow for all session numbers.
"""
from __future__ import annotations

from typing import Dict, Any

from annotate.utils.logger import log


def longitudinal_context_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich state with patient history context before extraction begins.

    If no patient_profile is present (session 1 or profile-less run),
    returns state unchanged.

    Returns partial state update:
      enriched_chat       — conversation + history header (used by extractors)
      longitudinal_flags  — list of active warnings for downstream agents
    """
    profile = state.get("patient_profile")

    if profile is None or profile.is_first_session:
        log("longitudinal_context: no prior history — running as first session.")
        return {
            "enriched_chat": state.get("chat", ""),
            "longitudinal_flags": [],
        }

    log(f"longitudinal_context: injecting history for session "
        f"{profile.session_count + 1} (patient={profile.patient_id})")

    context = profile.context_summary
    raw_chat = state.get("chat", "")

    # Build enriched conversation text — history header + separator + current chat
    enriched = (
        f"{context}\n"
        f"{'='*60}\n"
        f"CURRENT SESSION CONVERSATION:\n"
        f"{raw_chat}"
    )

    # Compute active longitudinal warning flags
    flags = _compute_flags(profile)
    if flags:
        log(f"longitudinal_context: active flags → {flags}")

    return {
        "enriched_chat": enriched,
        "longitudinal_flags": flags,
    }


def _compute_flags(profile) -> list[str]:
    """Generate warning flags from profile state for downstream agents."""
    flags = []

    if profile.stagnation_count >= 3:
        flags.append(
            f"STAGNATION: patient has been '{profile.current_stage}' for "
            f"{profile.stagnation_count} consecutive sessions"
        )

    if profile.relapse_count > 0:
        flags.append(
            f"RELAPSE_HISTORY: {profile.relapse_count} relapse(s) in history — "
            f"watch for relapse indicators in current session"
        )

    if profile.current_stage in ("action", "maintenance") and profile.relapse_count > 0:
        flags.append(
            "HIGH_RELAPSE_RISK: patient in post-action stage with prior relapse(s)"
        )

    if profile.stage_direction_history and profile.stage_direction_history[-1] == "regression":
        flags.append(
            f"RECENT_REGRESSION: last session showed regression from "
            f"'{profile.previous_stage}' to '{profile.current_stage}'"
        )

    if len(profile.persistent_triggers) > 3:
        flags.append(
            f"PERSISTENT_TRIGGERS: {len(profile.persistent_triggers)} triggers "
            f"unresolved across multiple sessions"
        )

    return flags
