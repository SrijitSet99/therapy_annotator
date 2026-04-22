"""
schemas/longitudinal_schema.py
--------------------------------
Schemas for multi-session longitudinal coherence.

Three layers:
  SessionRecord      — immutable snapshot of one session's annotation
  SessionDelta       — what changed between session N-1 and session N
  PatientProfile     — mutable longitudinal state, derived from all SessionRecords

Design principle: SessionRecords are NEVER mutated after being written.
The PatientProfile is always recomputable from the raw session records.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime


# ---------------------------------------------------------------------------
# Stage ordering — used for trajectory direction computation
# ---------------------------------------------------------------------------

STAGE_ORDER: Dict[str, int] = {
    "precontemplation": 0,
    "contemplation":    1,
    "preparation":      2,
    "action":           3,
    "maintenance":      4,
    "relapse":         -1,   # special: not on the linear scale
    "unknown":         -2,
}

VALID_STAGES = set(STAGE_ORDER.keys())


def stage_direction(before: str, after: str) -> str:
    """
    Classify the direction of a stage transition.

    Returns one of:
      'progress'    — moved forward on TTM scale
      'regression'  — moved backward (not relapse)
      'relapse'     — explicitly reached relapse stage
      'recovery'    — moved forward from relapse
      'stable'      — no change
      'unknown'     — one or both stages unclassifiable
    """
    if "unknown" in (before, after):
        return "unknown"
    if after == "relapse":
        return "relapse"
    if before == "relapse":
        return "recovery"
    b = STAGE_ORDER.get(before, -2)
    a = STAGE_ORDER.get(after, -2)
    if b == -2 or a == -2:
        return "unknown"
    if a > b:
        return "progress"
    if a < b:
        return "regression"
    return "stable"


# ---------------------------------------------------------------------------
# SessionRecord — one session's annotation, immutable after creation
# ---------------------------------------------------------------------------

class SessionRecord(BaseModel):
    """
    Immutable snapshot of a single counseling session annotation.
    Created from a DatasetRow after the pipeline completes.
    Never modified after being written to the PatientProfile.
    """
    session_number: int
    session_date: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    conversation_id: Optional[str] = None

    # Core annotation fields (mirrors DatasetRow)
    stage_of_change: str = "unknown"
    main_concern: str = "unknown"
    primary_trigger: str = "unknown"
    quit_attempt_history: str = "unknown"
    recommended_intervention: List[str] = Field(default_factory=list)
    reasoning_summary: Optional[str] = None
    confidence_score: Optional[float] = None

    # All extracted signals (for longitudinal diffing)
    all_triggers: List[str] = Field(default_factory=list)
    all_concerns: List[str] = Field(default_factory=list)

    # Pipeline metadata
    debate_rounds_used: Optional[int] = None
    vote_agreement: Optional[float] = None
    sanity_passed: Optional[bool] = None
    inconsistencies: List[str] = Field(default_factory=list)

    # Raw conversation text (for context injection into future sessions)
    conversation_excerpt: Optional[str] = None  # first 500 chars


# ---------------------------------------------------------------------------
# SessionDelta — what changed between consecutive sessions
# ---------------------------------------------------------------------------

class SessionDelta(BaseModel):
    """
    Computed difference between session N-1 and session N.
    Surfaced in the DatasetRow and used by the longitudinal sanity checker.
    """
    from_session: int
    to_session: int
    from_date: Optional[str] = None
    to_date: Optional[str] = None

    # Stage change
    stage_before: str = "unknown"
    stage_after: str = "unknown"
    stage_changed: bool = False
    stage_direction: str = "unknown"   # progress|regression|relapse|recovery|stable

    # Signal evolution
    new_triggers: List[str] = Field(default_factory=list)
    resolved_triggers: List[str] = Field(default_factory=list)
    persistent_triggers: List[str] = Field(default_factory=list)

    new_concerns: List[str] = Field(default_factory=list)
    resolved_concerns: List[str] = Field(default_factory=list)
    persistent_concerns: List[str] = Field(default_factory=list)

    # Intervention tracking
    new_interventions: List[str] = Field(default_factory=list)
    continuing_interventions: List[str] = Field(default_factory=list)
    dropped_interventions: List[str] = Field(default_factory=list)

    # Clinical flags
    relapse_detected: bool = False
    rapid_progress_detected: bool = False    # skipped >1 stage
    stagnation_detected: bool = False        # stable for 3+ sessions
    concern_shift_detected: bool = False     # main concern changed significantly
    trigger_resolved: bool = False           # at least one trigger no longer present

    # Narrative summary produced by delta_agent
    clinical_summary: str = ""

    # Confidence in this delta (how reliable is the comparison)
    delta_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# PatientProfile — mutable longitudinal state across all sessions
# ---------------------------------------------------------------------------

class PatientProfile(BaseModel):
    """
    Persistent longitudinal state for a single patient across all sessions.
    Recomputable from sessions list at any time.
    Stored as JSON in data/patients/{patient_id}.json
    """
    patient_id: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # Ordered session history (immutable once appended)
    sessions: List[SessionRecord] = Field(default_factory=list)

    # Latest delta (updated after each session)
    latest_delta: Optional[SessionDelta] = None

    # Derived longitudinal fields (recomputed after each new session)
    stage_trajectory: List[str] = Field(default_factory=list)
    stage_direction_history: List[str] = Field(default_factory=list)

    # Cumulative signal sets
    all_triggers_ever: List[str] = Field(default_factory=list)
    all_concerns_ever: List[str] = Field(default_factory=list)
    interventions_tried: List[str] = Field(default_factory=list)

    # Persistence flags (present in most recent N sessions)
    persistent_triggers: List[str] = Field(default_factory=list)    # last 2+ sessions
    persistent_concerns: List[str] = Field(default_factory=list)    # last 2+ sessions
    resolved_triggers: List[str] = Field(default_factory=list)      # not in last 2 sessions

    # Clinical summary state
    total_sessions: int = 0
    stagnation_count: int = 0      # consecutive sessions with no stage change
    relapse_count: int = 0
    best_stage_reached: str = "unknown"

    # Context string injected into extractors (recomputed each session)
    context_summary: str = ""

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> str:
        if not self.sessions:
            return "unknown"
        return self.sessions[-1].stage_of_change

    @property
    def previous_stage(self) -> str:
        if len(self.sessions) < 2:
            return "unknown"
        return self.sessions[-2].stage_of_change

    @property
    def session_count(self) -> int:
        return len(self.sessions)

    @property
    def is_first_session(self) -> bool:
        return len(self.sessions) == 0

    def last_n_stages(self, n: int = 3) -> List[str]:
        return [s.stage_of_change for s in self.sessions[-n:]]

    def get_session(self, n: int) -> Optional[SessionRecord]:
        """Get session by 1-based session number."""
        for s in self.sessions:
            if s.session_number == n:
                return s
        return None


# ---------------------------------------------------------------------------
# LongitudinalAnnotation — final output for multi-session pipeline
# ---------------------------------------------------------------------------

class LongitudinalAnnotation(BaseModel):
    """
    Complete output for a multi-session pipeline run.
    Combines the current session's DatasetRow with longitudinal context.
    """
    patient_id: str
    session_number: int

    # Current session annotation
    current_annotation: Dict[str, Any] = Field(default_factory=dict)

    # Longitudinal context
    delta: Optional[SessionDelta] = None
    stage_trajectory: List[str] = Field(default_factory=list)
    persistent_triggers: List[str] = Field(default_factory=list)
    persistent_concerns: List[str] = Field(default_factory=list)
    interventions_tried: List[str] = Field(default_factory=list)

    # Longitudinal flags
    relapse_detected: bool = False
    stagnation_detected: bool = False
    overall_progress: str = "unknown"   # improving|stable|regression|relapse

    # Longitudinal sanity issues (cross-session)
    longitudinal_flags: List[str] = Field(default_factory=list)
