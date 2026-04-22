from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any

# ---------------------------------------------------------------------------
# Extractor outputs
# ---------------------------------------------------------------------------

class AdviceOutput(BaseModel):
    advice: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


class AttemptOutput(BaseModel):
    attempts: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)

class ReadinessOutput(BaseModel):
    readiness_stage: str = Field(default="unknown")
    evidence: Optional[str] = None

    @field_validator("readiness_stage")
    @classmethod
    def validate_stage(cls, v: str) -> str:
        allowed = {"precontemplation", "contemplation", "preparation",
                   "action", "maintenance", "relapse", "unknown"}
        normalized = v.strip().lower()
        return normalized if normalized in allowed else "unknown"


class ConcernOutput(BaseModel):
    concerns: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


class TriggerOutput(BaseModel):
    triggers: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluation & Confidence
# ---------------------------------------------------------------------------

class ConfidenceResult(BaseModel):
    """The output of the external judge verifying an extraction."""
    confidence: float = Field(..., ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    severity: str = Field(default="none")


# ---------------------------------------------------------------------------
# Review / Revision (Used by advice/trigger review agents)
# ---------------------------------------------------------------------------

class ReviewOutput(BaseModel):
    is_accurate: bool = True
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    severity: str = Field(default="low")

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        allowed = {"none", "low", "medium", "high"}
        return v.strip().lower() if v.strip().lower() in allowed else "low"


# ---------------------------------------------------------------------------
# Debate (Proposer-Critic-Reviser loop)
# ---------------------------------------------------------------------------

class DebateRound(BaseModel):
    round_number: int
    proposer_output: Dict[str, Any] = Field(default_factory=dict)
    critic_feedback: str = ""
    revised_output: Dict[str, Any] = Field(default_factory=dict)
    agreement_reached: bool = False


class DebateResult(BaseModel):
    agent_name: str = ""
    rounds: List[DebateRound] = Field(default_factory=list)
    final_output: Dict[str, Any] = Field(default_factory=dict)
    total_rounds: int = 0
    converged: bool = False


# ---------------------------------------------------------------------------
# Multi-voter Consensus & Arbitration
# ---------------------------------------------------------------------------

class ConsensusVote(BaseModel):
    voter_id: str
    stage_of_change: str = "unknown"
    main_concern: str = "unknown"
    primary_trigger: str = "unknown"
    quit_attempt_history: str = "unknown"
    recommended_intervention: List[str] = Field(default_factory=list)


class ArbitrationResult(BaseModel):
    stage_of_change: str = "unknown"
    main_concern: str = "unknown"
    primary_trigger: str = "unknown"
    quit_attempt_history: str = "unknown"
    recommended_intervention: List[str] = Field(default_factory=list)
    vote_agreement: float = Field(default=0.0, ge=0.0, le=1.0)
    dissenting_fields: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sanity check (Post-Consensus validation)
# ---------------------------------------------------------------------------

class SanityCheckResult(BaseModel):
    passed: bool = True
    inconsistencies: List[str] = Field(default_factory=list)
    severity: str = "none"


# ---------------------------------------------------------------------------
# Final dataset row (The output of reasoning_agent.py)
# ---------------------------------------------------------------------------

class DatasetRow(BaseModel):
    # Clinical Fields
    stage_of_change: str = Field(default="unknown")
    main_concern: str = Field(default="unknown")
    primary_trigger: str = Field(default="unknown")
    quit_attempt_history: str = Field(default="unknown")
    recommended_intervention: List[str] = Field(default_factory=list)
    
    # Reasoning & Quality Metadata
    reasoning_summary: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Research Metadata (Longitudinal/MAS Trace)
    debate_rounds_used: int = 0
    vote_agreement: float = Field(default=0.0, ge=0.0, le=1.0)
    sanity_passed: bool = True
    inconsistencies: List[str] = Field(default_factory=list)
