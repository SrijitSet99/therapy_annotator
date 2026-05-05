"""
graph/runner.py
----------------
Public API for single-session and multi-session pipeline runs.

run_pipeline()           — single session, no longitudinal tracking
run_session()            — multi-session, full longitudinal coherence
run_patient_sessions()   — convenience wrapper to run all sessions for a patient

`enable_debate=True` lifts the workflow's confidence/revision loop to use
DEBATE_CONFIDENCE_THRESHOLD and DEBATE_MAX_REVISION_ROUNDS — i.e. it runs
longer with a stricter acceptance bar. There is no separate per-extractor
debate loop.
"""
from typing import Dict, Any, Optional, List

from annotate.graph.workflow import build_graph
from annotate.agents.memory.patient_memory import PatientMemory
from annotate.agents.memory.agent_memory import AgentMemory
from annotate.config.settings import (
    CONFIDENCE_DEBATE_THRESHOLD, MAX_REVISION_ROUNDS,
    DEBATE_CONFIDENCE_THRESHOLD, DEBATE_MAX_REVISION_ROUNDS,
)
from annotate.utils.logger import log

_graph = None


def _get_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _debate_config(enable_debate: bool) -> Dict[str, Any]:
    if enable_debate:
        return {
            "confidence_threshold":  DEBATE_CONFIDENCE_THRESHOLD,
            "max_revision_rounds":   DEBATE_MAX_REVISION_ROUNDS,
        }
    return {
        "confidence_threshold":  CONFIDENCE_DEBATE_THRESHOLD,
        "max_revision_rounds":   MAX_REVISION_ROUNDS,
    }


# ---------------------------------------------------------------------------
# Single-session (original API — unchanged for backwards compatibility)
# ---------------------------------------------------------------------------

def run_pipeline(
    chat: str,
    conversation_id: Optional[str] = None,
    memory: Optional[AgentMemory] = None,
    enable_debate: bool = False,
) -> Dict[str, Any]:
    """
    Run a single conversation through the pipeline without longitudinal tracking.
    """
    log(f"run_pipeline: starting{('  id=' + conversation_id) if conversation_id else ''}"
        f"{' (debate mode)' if enable_debate else ''}")

    initial_state: Dict[str, Any] = {
        "chat": chat,
        "enriched_chat": "",
        "patient_profile": None,
        "longitudinal_flags": [],
        "advice": {}, "attempt": {}, "readiness": {}, "concern": {}, "triggers": {},
        "consensus": {}, "confidence_results": {},
        "sanity_check": {}, "longitudinal_sanity": {},
        "revision_count": 0,           # initialise loop counter
        "session_delta": None,
        "dataset_row": {}, "longitudinal_output": {},
        **_debate_config(enable_debate),
    }
    if conversation_id:
        initial_state["conversation_id"] = conversation_id

    result = _get_graph().invoke(initial_state)
    row = result.get("dataset_row", {})

    if memory and conversation_id:
        memory.store_case(conversation_id, row)

    log("run_pipeline: complete.")
    return row


# ---------------------------------------------------------------------------
# Multi-session (new longitudinal API)
# ---------------------------------------------------------------------------

def run_session(
    chat: str,
    patient_id: str,
    conversation_id: Optional[str] = None,
    patient_memory: Optional[PatientMemory] = None,
    enable_debate: bool = False,
) -> Dict[str, Any]:
    """
    Run a counseling session with full longitudinal coherence.
    """
    pm = patient_memory or PatientMemory()
    profile = pm.load_or_create(patient_id)

    session_number = profile.session_count + 1
    log(f"run_session: patient='{patient_id}' session={session_number}")

    initial_state: Dict[str, Any] = {
        "chat": chat,
        "enriched_chat": "",
        "patient_profile": profile,
        "longitudinal_flags": [],
        "advice": {}, "attempt": {}, "readiness": {}, "concern": {}, "triggers": {},
        "consensus": {}, "confidence_results": {},
        "sanity_check": {}, "longitudinal_sanity": {},
        "revision_count": 0,           # initialise loop counter
        "session_delta": None,
        "dataset_row": {}, "longitudinal_output": {},
        **_debate_config(enable_debate),
    }
    if conversation_id:
        initial_state["conversation_id"] = conversation_id

    result = _get_graph().invoke(initial_state)

    log(f"run_session: complete — session {session_number} for '{patient_id}'")

    return {
        "dataset_row":         result.get("dataset_row", {}),
        "longitudinal_output": result.get("longitudinal_output", {}),
        "session_number":      session_number,
        "patient_id":          patient_id,
    }


def run_patient_sessions(
    sessions: List[Dict[str, Any]],
    patient_id: str,
    patient_memory: Optional[PatientMemory] = None,
    enable_debate: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper to run multiple sessions for the same patient sequentially.
    """
    pm = patient_memory or PatientMemory()
    results = []
    for i, item in enumerate(sessions, 1):
        conv = item.get("conversation", item.get("chat", ""))
        conv_id = item.get("id", f"{patient_id}_session_{i:03d}")
        result = run_session(conv, patient_id=patient_id,
                             conversation_id=conv_id, patient_memory=pm,
                             enable_debate=enable_debate)
        results.append(result)
        log(f"run_patient_sessions: completed {i}/{len(sessions)} for '{patient_id}'")
    return results