from typing import TypedDict, Dict, Any, List, Optional


class GraphState(TypedDict, total=False):
    """
    Shared mutable state for the full pipeline (single + multi-session).
    """
    # ── Input ─────────────────────────────────────────────────────────
    chat: str                    # raw conversation text
    enriched_chat: str           # chat + longitudinal history header (sessions 2+)

    # ── Longitudinal context (injected before extraction) ─────────────
    patient_profile: Any         # PatientProfile object (optional)
    longitudinal_flags: List[str]  # active warnings from context_agent

    # ── Extractor outputs ─────────────────────────────────────────────
    advice:    Dict[str, Any]
    attempt:   Dict[str, Any]
    readiness: Dict[str, Any]
    concern:   Dict[str, Any]
    triggers:  Dict[str, Any]

    # ── Multi-voter consensus ──────────────────────────────────────────
    consensus: Dict[str, Any]

    # ── Post-hoc validation ───────────────────────────────────────────
    confidence_results: Dict[str, Any]
    sanity_check:        Dict[str, Any]
    longitudinal_sanity: Dict[str, Any]

    # ── Confidence → revision loop metadata ───────────────────────────
    revision_count: int          # how many revision cycles have run so far

    # ── Longitudinal delta ────────────────────────────────────────────
    session_delta: Optional[Dict[str, Any]]

    # ── Final outputs ─────────────────────────────────────────────────
    dataset_row:         Dict[str, Any]
    longitudinal_output: Dict[str, Any]

    # ── Pipeline metadata ─────────────────────────────────────────────
    conversation_id:            Optional[str]
    cross_signal_disagreement:  Optional[float]