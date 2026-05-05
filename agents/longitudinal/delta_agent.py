"""
agents/longitudinal/delta_agent.py
------------------------------------
Runs AFTER the reasoning agent on sessions 2+.

Computes the SessionDelta between the just-completed session N and
the previous session N-1, then calls the LLM to produce a clinical
narrative summary of the changes.

Delta computation is split into two parts:
  1. Structural diff — pure Python, deterministic (triggers, concerns,
     interventions, stage direction)
  2. Clinical narrative — one LLM call that interprets the structural
     diff in clinical language

This separation means the structural facts are always correct even if
the LLM narrative call fails.
"""
from __future__ import annotations

from typing import Dict, Any, List, Set, Tuple

from annotate.llm.ollama_client import call_ollama
from annotate.llm.json_parser import repair_json
from annotate.utils.logger import log, warning, log_data
from annotate.utils.prompt_loader import format_prompt, load_prompt
from annotate.schemas.longitudinal_schema import SessionDelta, stage_direction, STAGE_ORDER


def delta_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the SessionDelta between session N-1 and the current session.

    If no prior session exists (first session), returns empty delta.
    Injects the delta into state under 'session_delta'.
    """
    profile = state.get("patient_profile")
    current_row = state.get("dataset_row", {})

    if profile is None or profile.session_count == 0:
        log("delta_agent: no prior session — skipping delta computation.")
        return {"session_delta": None}

    prev = profile.sessions[-1]     # last completed session (N-1)
    n = profile.session_count + 1   # current session number

    log(f"delta_agent: computing delta between session {prev.session_number} and {n}")

    # ── 1. Structural diff — stage ────────────────────────────────────

    stage_before = prev.stage_of_change
    stage_after  = current_row.get("stage_of_change", "unknown")

    # FIX 2: was `=` (assignment); must be `==` (comparison).
    # FIX 3: logic was inverted — direction is computed in both branches,
    #         but stage fallback only applies when stage_after is unknown.
    if stage_after == "unknown":
        stage_after = stage_before

    direction = stage_direction(stage_before, stage_after)

    # ── 2. Trigger diff with semantic equivalence + explicit resolution ──

    # FIX 9: removed stray bare `Modify` token that caused NameError.
    prev_triggers_raw = list(_normalise_set(prev.all_triggers))
    curr_triggers_raw = list(_normalise_set(state.get("triggers", {}).get("triggers", [])))
    transcript        = state.get("chat", "")

    # FIX 4 & 5: was calling non-existent function with misaligned args.
    new_triggers, persistent_triggers, resolved_triggers = _compute_delta(
        signal_type=  "trigger",
        prev=         prev_triggers_raw,
        curr=         curr_triggers_raw,
        transcript=   transcript,
    )

    # ── 3. Concern diff with semantic equivalence + explicit resolution ──

    # FIX 9: removed stray bare `Modify` token.
    prev_concerns_raw = list(_normalise_set(prev.all_concerns))
    curr_concerns_raw = list(_normalise_set(state.get("concern", {}).get("concerns", [])))

    # FIX 5: was calling non-existent `_compute_concern_delta`; unified to `_compute_delta`.
    new_concerns, persistent_concerns, resolved_concerns = _compute_delta(
        signal_type= "concern",
        prev=        prev_concerns_raw,
        curr=        curr_concerns_raw,
        transcript=  transcript,
    )

    # ── 4. Intervention diff (no resolution check — transcript not needed) ──

    # FIX 9: removed stray bare `Modify` token.
    prev_interventions_raw = list(_normalise_set(prev.recommended_intervention))
    curr_interventions_raw = list(_normalise_set(current_row.get("recommended_intervention", [])))

    # FIX 5: was calling non-existent `_compute_intervention_delta`; unified to `_compute_delta`.
    # FIX 8: interventions have no transcript-based resolution check; pass empty string.
    new_interventions, continuing_interventions, dropped_interventions = _compute_delta(
        signal_type= "intervention",
        prev=        prev_interventions_raw,
        curr=        curr_interventions_raw,
        transcript=  "",
    )

    # ── 5. Clinical flags ─────────────────────────────────────────────

    relapse_detected = (stage_after == "relapse")

    before_order = STAGE_ORDER.get(stage_before, -2)
    after_order  = STAGE_ORDER.get(stage_after,  -2)
    rapid_progress = (
        direction == "progress"
        and before_order >= 0
        and after_order >= 0
        and (after_order - before_order) > 1
    )

    stagnation = (direction == "stable" and profile.stagnation_count >= 2)

    # Concern shift: any concern added or resolved between the two sessions.
    concern_shift = bool(new_concerns or resolved_concerns)

    # ── 6. LLM narrative ─────────────────────────────────────────────

    clinical_summary = ""
    delta_confidence = 0.7

    try:
        prompt = format_prompt(
            load_prompt("longitudinal.delta_narrative"),
            n_minus_1=prev.session_number,
            n=n,
            stage_before=stage_before,
            stage_after=stage_after,
            concerns_before=", ".join(prev.all_concerns) or "none",
            concerns_after=", ".join(current_row.get("all_concerns", [])) or "none",
            triggers_before=", ".join(prev.all_triggers) or "none",
            triggers_after=", ".join(current_row.get("all_triggers", [])) or "none",
            interventions_before=", ".join(prev.recommended_intervention) or "none",
            interventions_after=", ".join(current_row.get("recommended_intervention", [])) or "none",
            stage_direction=direction,
            new_triggers=", ".join(new_triggers) or "none",
            resolved_triggers=", ".join(resolved_triggers) or "none",
            new_concerns=", ".join(new_concerns) or "none",
            resolved_concerns=", ".join(resolved_concerns) or "none",
            new_interventions=", ".join(new_interventions) or "none",
        )
        raw = call_ollama(prompt)
        parsed = repair_json(raw)
        clinical_summary = parsed.get("clinical_summary", "")
        delta_confidence = float(parsed.get("delta_confidence", 0.7))
    except Exception as exc:
        warning(f"delta_agent: narrative call failed — {exc}. Using structural summary.")
        clinical_summary = _fallback_narrative(
            stage_before, stage_after, direction,
            new_triggers, resolved_triggers, new_concerns
        )

    # ── 7. Build SessionDelta ─────────────────────────────────────────

    delta = SessionDelta(
        from_session=prev.session_number,
        to_session=n,
        from_date=prev.session_date,
        stage_before=stage_before,
        stage_after=stage_after,
        stage_changed=(stage_before != stage_after),
        stage_direction=direction,
        new_triggers=new_triggers,
        resolved_triggers=resolved_triggers,
        persistent_triggers=persistent_triggers,
        new_concerns=new_concerns,
        resolved_concerns=resolved_concerns,
        persistent_concerns=persistent_concerns,
        new_interventions=new_interventions,
        continuing_interventions=continuing_interventions,
        dropped_interventions=dropped_interventions,
        relapse_detected=relapse_detected,
        rapid_progress_detected=rapid_progress,
        stagnation_detected=stagnation,
        concern_shift_detected=concern_shift,
        trigger_resolved=len(resolved_triggers) > 0,
        clinical_summary=clinical_summary,
        delta_confidence=min(1.0, max(0.0, delta_confidence)),
    )

    log(
        f"delta_agent: stage {stage_before} → {stage_after} ({direction}) | "
        f"new_triggers={new_triggers} | resolved={resolved_triggers} | "
        f"relapse={relapse_detected}"
    )

    payload = {"session_delta": delta.model_dump()}
    log_data("delta_agent.output", payload)
    return payload


# ---------------------------------------------------------------------------
# Generic signal delta (triggers, concerns, interventions)
# ---------------------------------------------------------------------------

def _compute_delta(
    signal_type: str,
    prev: List[str],
    curr: List[str],
    transcript: str,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (new_items, persistent_items, resolved_items).

    Step 1 — Semantic diff: ask the LLM which current items are genuinely
              new vs. semantically equivalent to previous ones.
    Step 2 — Resolution check: ask the LLM to scan the transcript and flag
              only those previous items the patient explicitly said are gone.
              Skipped when transcript is empty (e.g. interventions).

    Falls back to naive set-difference if either LLM call fails.
    """
    if not prev and not curr:
        return [], [], []

    # ── Step 1: semantic equivalence diff ────────────────────────────
    new: List[str] = []
    persistent: List[str] = []

    try:
        sem_prompt = format_prompt(
            load_prompt("longitudinal.semantic_diff"),
            signal=    signal_type,
            prev_items="\n".join(f"- {t}" for t in prev) or "(none)",
            curr_items="\n".join(f"- {t}" for t in curr) or "(none)",
        )
        raw = call_ollama(sem_prompt)
        parsed = repair_json(raw)
        # FIX 6: was using `prev_triggers`/`curr_triggers`; now uses the generic params.
        new       = [t.strip().lower() for t in parsed.get(f"new_{signal_type}s", []) if t.strip()]
        persistent= [t.strip().lower() for t in parsed.get(f"persistent_{signal_type}s", []) if t.strip()]
        log(f"_compute_delta({signal_type}): semantic diff → new={new}, persistent={persistent}")
    except Exception as exc:
        warning(f"_compute_delta({signal_type}): semantic diff failed — {exc}. Falling back to set diff.")
        prev_set = set(prev)
        curr_set = set(curr)
        new       = sorted(curr_set - prev_set)
        persistent= sorted(curr_set & prev_set)

    # ── Step 2: explicit resolution check ────────────────────────────
    resolved: List[str] = []

    if prev and transcript:
        try:
            res_prompt = format_prompt(
                load_prompt("longitudinal.resolved_validation"),
                signal=    signal_type,
                prev_items="\n".join(f"- {t}" for t in prev),
                transcript=transcript[:6000],   # guard against huge transcripts
            )
            raw = call_ollama(res_prompt)
            parsed = repair_json(raw)
            resolved = [
                t.strip().lower()
                for t in parsed.get(f"resolved_{signal_type}s", [])
                if t.strip()
            ]
            log(f"_compute_delta({signal_type}): resolution check → resolved={resolved}")
        except Exception as exc:
            warning(f"_compute_delta({signal_type}): resolution check failed — {exc}. None marked resolved.")
            resolved = []
    else:
        log(f"_compute_delta({signal_type}): no transcript — skipping resolution check.")

    return sorted(new), sorted(persistent), sorted(resolved)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _normalise_set(items: List[str]) -> Set[str]:
    return {item.strip().lower() for item in items if item.strip()}


def _fallback_narrative(
    stage_before: str,
    stage_after: str,
    direction: str,
    new_triggers: List[str],
    resolved_triggers: List[str],
    new_concerns: List[str],
) -> str:
    parts = []
    if stage_before != stage_after:
        parts.append(
            f"Stage changed from {stage_before} to {stage_after} ({direction})."
        )
    else:
        parts.append(f"Stage remained at {stage_after}.")
    if new_triggers:
        parts.append(f"New triggers identified: {', '.join(new_triggers)}.")
    if resolved_triggers:
        parts.append(f"Patient explicitly resolved the following triggers: {', '.join(resolved_triggers)}.")
    if new_concerns:
        parts.append(f"New concerns emerged: {', '.join(new_concerns)}.")
    return " ".join(parts) or "No significant changes detected between sessions."
