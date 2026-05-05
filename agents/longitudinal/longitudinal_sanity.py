"""
agents/longitudinal/longitudinal_sanity.py
--------------------------------------------
Cross-session consistency checker.

Extends the within-session sanity_check_agent with rules that span
multiple sessions. Runs after delta_agent has computed the SessionDelta.

Rules checked:
  1. Stage skip > 1 step forward in one session (likely extraction error)
  2. Relapse detected but no mention of relapse in conversation
  3. Previously failed intervention recommended again without acknowledgment
  4. Stagnation: 4+ consecutive stable sessions (clinically significant flag)
  5. Rapid regression: action/maintenance → precontemplation in one session
  6. Conflicting concern: a current concern contradicts patient's stated stage
  7. Trigger persistence: critical trigger present for 3+ sessions unchanged
"""
from __future__ import annotations

from typing import Dict, Any, List

from annotate.utils.logger import log, log_data
from annotate.schemas.longitudinal_schema import STAGE_ORDER


def longitudinal_sanity_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check cross-session consistency.
    Returns partial state update: 'longitudinal_sanity' dict.
    """
    profile     = state.get("patient_profile")
    delta       = state.get("session_delta")
    current_row = state.get("dataset_row", {})
    conversation = state.get("chat", "").lower()

    if profile is None or profile.session_count == 0 or delta is None:
        log("longitudinal_sanity: first session — skipping cross-session checks.")
        payload = {"longitudinal_sanity": {"passed": True, "flags": [], "severity": "none"}}
        log_data("longitudinal_sanity.output", payload)
        return payload

    flags: List[str] = []

    # ── Rule 1: Stage skip > 1 step ───────────────────────────────────
    before_order = STAGE_ORDER.get(delta.get("stage_before", "unknown"), -2)
    after_order  = STAGE_ORDER.get(delta.get("stage_after",  "unknown"), -2)
    if (before_order >= 0 and after_order >= 0
            and (after_order - before_order) > 1):
        flags.append(
            f"STAGE_SKIP: jumped from '{delta['stage_before']}' to "
            f"'{delta['stage_after']}' in one session (skipped "
            f"{after_order - before_order - 1} stage(s)). "
            f"Verify extraction accuracy."
        )

    # ── Rule 2: Relapse detected but no relapse language in chat ──────
    if delta.get("relapse_detected"):
        relapse_keywords = ["relapse", "started smoking again", "gave up", "went back",
                            "smoking again", "failed", "slipped"]
        mentioned = any(kw in conversation for kw in relapse_keywords)
        if not mentioned:
            flags.append(
                "SILENT_RELAPSE: stage annotated as 'relapse' but no relapse "
                "language found in conversation. Check extraction."
            )

    # ── Rule 3: Previously failed intervention re-recommended ─────────
    # (Simple heuristic: if an intervention was tried in session 1 and
    #  recommended again in session 3+ without the word "again" or "continue")
    if profile.session_count >= 2:
        tried_set = {i.lower().strip() for i in profile.interventions_tried}
        current_new = [i for i in delta.get("new_interventions", [])
                       if i.lower().strip() in tried_set]
        if current_new:
            flags.append(
                f"REPEATED_INTERVENTION: {current_new} was previously tried "
                f"and is being recommended again. Consider noting outcome of prior use."
            )

    # ── Rule 4: Stagnation ≥ 4 consecutive stable sessions ───────────
    if profile.stagnation_count >= 4:
        flags.append(
            f"PROLONGED_STAGNATION: patient has been in '{profile.current_stage}' "
            f"for {profile.stagnation_count} consecutive sessions. "
            f"Consider escalating or changing approach."
        )

    # ── Rule 5: Rapid regression from high stage ──────────────────────
    if (before_order >= STAGE_ORDER.get("action", 3)
            and after_order <= STAGE_ORDER.get("contemplation", 1)
            and after_order >= 0):
        flags.append(
            f"SEVERE_REGRESSION: dropped from '{delta['stage_before']}' to "
            f"'{delta['stage_after']}' in one session. Assess for crisis or "
            f"significant life event."
        )

    # ── Rule 6: Trigger persistent for 3+ sessions without intervention ──
    sessions_with_trigger: Dict[str, int] = {}
    for session in profile.sessions:
        for t in session.all_triggers:
            key = t.lower().strip()
            sessions_with_trigger[key] = sessions_with_trigger.get(key, 0) + 1

    stuck_triggers = [t for t, count in sessions_with_trigger.items() if count >= 3]
    current_interventions_lower = {
        i.lower() for i in current_row.get("recommended_intervention", [])
    }
    for stuck in stuck_triggers:
        # Check if any current intervention addresses this trigger
        addressed = any(stuck in iv or iv in stuck
                        for iv in current_interventions_lower)
        if not addressed:
            flags.append(
                f"UNADDRESSED_PERSISTENT_TRIGGER: '{stuck}' has been present for "
                f"{sessions_with_trigger[stuck]} sessions without a targeted intervention."
            )

    # ── Rule 7: Concern contradicts stage ────────────────────────────
    stage = current_row.get("stage_of_change", "unknown")
    concerns_lower = [c.lower() for c in current_row.get("all_concerns", [])]
    not_ready_phrases = ["not ready", "don't want", "not considering", "no interest"]
    if stage in ("action", "maintenance"):
        for concern in concerns_lower:
            if any(phrase in concern for phrase in not_ready_phrases):
                flags.append(
                    f"CONCERN_STAGE_MISMATCH: stage is '{stage}' but a concern suggests "
                    f"ambivalence: '{concern}'"
                )
                break

    # ── Severity ──────────────────────────────────────────────────────
    passed = len(flags) == 0
    severity = "none" if passed else ("high" if len(flags) >= 3 else "medium")

    if flags:
        log(f"longitudinal_sanity: {len(flags)} flag(s) (severity={severity})")
        for f in flags:
            log(f"  FLAG: {f}")
    else:
        log("longitudinal_sanity: all cross-session checks passed.")

    payload = {
        "longitudinal_sanity": {
            "passed": passed,
            "flags": flags,
            "severity": severity,
        }
    }
    log_data("longitudinal_sanity.output", payload)
    return payload
