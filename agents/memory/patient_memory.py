"""
agents/memory/patient_memory.py
---------------------------------
Persistent per-patient storage for longitudinal coherence.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from annotate.schemas.longitudinal_schema import (
    PatientProfile, SessionRecord, SessionDelta,
    STAGE_ORDER, stage_direction,
)
from annotate.config.settings import PATIENTS_DIR
from annotate.utils.logger import log

_DEFAULT_PATIENTS_DIR = Path(PATIENTS_DIR)


# ---------------------------------------------------------------------------
# 🔧 Normalization Utilities (CRITICAL FIX)
# ---------------------------------------------------------------------------

def _normalize_signal_list(items: Optional[List[Any]]) -> List[str]:
    """
    Normalize mixed list[str | dict] → list[str]

    Supports:
    - {"trigger_type": "...", "confidence": 0.8}
    - {"concern_type": "..."}
    - {"text": "..."}
    - plain strings

    Always returns clean List[str]
    """
    if not items:
        return []

    normalized: List[str] = []

    for x in items:
        if isinstance(x, dict):
            val = (
                x.get("trigger_type")
                or x.get("concern_type")
                or x.get("text")
                or x.get("label")
                or str(x)
            )
        else:
            val = str(x)

        val = val.strip()
        if val:
            normalized.append(val)

    return normalized


def _dedup(lst: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in lst:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result


# ---------------------------------------------------------------------------
# Patient Memory
# ---------------------------------------------------------------------------

class PatientMemory:

    def __init__(self, patients_dir: Optional[str] = None) -> None:
        self.patients_dir = Path(patients_dir) if patients_dir else _DEFAULT_PATIENTS_DIR
        self.patients_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def load_or_create(self, patient_id: str) -> PatientProfile:
        path = self._path(patient_id)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            profile = PatientProfile(**data)
            log(f"Loaded patient '{patient_id}' ({profile.session_count} sessions)")
            return profile

        log(f"Creating new patient '{patient_id}'")
        return PatientProfile(patient_id=patient_id)

    def save(self, profile: PatientProfile) -> None:
        profile.last_updated = datetime.utcnow().isoformat()
        self._path(profile.patient_id).write_text(
            profile.model_dump_json(indent=2), encoding="utf-8"
        )

    def _path(self, patient_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in patient_id)
        return self.patients_dir / f"{safe_id}.json"

    # ------------------------------------------------------------------
    # Append Session
    # ------------------------------------------------------------------

    def append_session(self, profile: PatientProfile, record: SessionRecord) -> PatientProfile:
        record.session_number = profile.session_count + 1
        profile.sessions.append(record)
        profile.total_sessions = len(profile.sessions)

        self._recompute_trajectory(profile)
        self._recompute_signal_sets(profile)
        self._recompute_persistence(profile)
        self._recompute_best_stage(profile)
        self._recompute_stagnation(profile)
        self._rebuild_context_summary(profile)

        log(f"Session {record.session_number} added → {profile.stage_trajectory}")
        return profile

    # ------------------------------------------------------------------
    # Recompute logic
    # ------------------------------------------------------------------

    def _recompute_trajectory(self, profile: PatientProfile):
        profile.stage_trajectory = [s.stage_of_change for s in profile.sessions]

        profile.stage_direction_history = []
        for i in range(1, len(profile.sessions)):
            profile.stage_direction_history.append(
                stage_direction(
                    profile.sessions[i - 1].stage_of_change,
                    profile.sessions[i].stage_of_change
                )
            )

        profile.relapse_count = profile.stage_trajectory.count("relapse")

    def _recompute_signal_sets(self, profile: PatientProfile):
        all_t, all_c, all_i = [], [], []

        for s in profile.sessions:
            all_t.extend(s.all_triggers)
            all_c.extend(s.all_concerns)
            all_i.extend(s.recommended_intervention)

        profile.all_triggers_ever = _dedup(all_t)
        profile.all_concerns_ever = _dedup(all_c)
        profile.interventions_tried = _dedup(all_i)

    def _recompute_persistence(self, profile: PatientProfile):
        if len(profile.sessions) < 2:
            last_triggers = set(profile.sessions[-1].all_triggers) if profile.sessions else set()
            last_concerns = set(profile.sessions[-1].all_concerns) if profile.sessions else set()
        else:
            last_triggers = set(profile.sessions[-1].all_triggers) & set(profile.sessions[-2].all_triggers)
            last_concerns = set(profile.sessions[-1].all_concerns) & set(profile.sessions[-2].all_concerns)

        profile.persistent_triggers = sorted(last_triggers)
        profile.persistent_concerns = sorted(last_concerns)

        profile.resolved_triggers = sorted(
            set(profile.all_triggers_ever) - set(profile.sessions[-1].all_triggers)
        )

    def _recompute_best_stage(self, profile: PatientProfile):
        best_order = -2
        best = "unknown"

        for stage in profile.stage_trajectory:
            if stage == "relapse":
                continue

            order = STAGE_ORDER.get(stage, -2)
            if order > best_order:
                best_order = order
                best = stage

        profile.best_stage_reached = best

    def _recompute_stagnation(self, profile: PatientProfile):
        count = 0
        for d in reversed(profile.stage_direction_history):
            if d == "stable":
                count += 1
            else:
                break
        profile.stagnation_count = count

    def _rebuild_context_summary(self, profile: PatientProfile):
        if profile.session_count == 0:
            profile.context_summary = ""
            return

        lines = [
            f"PATIENT HISTORY ({profile.session_count} sessions):",
            f"Trajectory: {' → '.join(profile.stage_trajectory)}",
            f"Current stage: {profile.current_stage}",
            f"Best stage: {profile.best_stage_reached}",
        ]

        if profile.persistent_triggers:
            lines.append(f"Persistent triggers: {', '.join(profile.persistent_triggers)}")

        if profile.persistent_concerns:
            lines.append(f"Persistent concerns: {', '.join(profile.persistent_concerns)}")

        if profile.stagnation_count >= 2:
            lines.append(f"⚠ Stagnation: {profile.stagnation_count} sessions")

        if profile.relapse_count > 0:
            lines.append(f"⚠ Relapse count: {profile.relapse_count}")

        profile.context_summary = "\n".join(lines)

    # ------------------------------------------------------------------
    # ✅ FIXED: Session Record Builder
    # ------------------------------------------------------------------

    @staticmethod
    def build_session_record(
        dataset_row: Dict[str, Any],
        conversation: str,
        conversation_id: Optional[str] = None,
        all_triggers: Optional[List[Any]] = None,
        all_concerns: Optional[List[Any]] = None,
    ) -> SessionRecord:
        # Prefer caller-supplied lists (extractor outputs) when present;
        # otherwise fall back to the lists already on the dataset row.
        triggers = all_triggers if all_triggers is not None else dataset_row.get("all_triggers", [])
        concerns = all_concerns if all_concerns is not None else dataset_row.get("all_concerns", [])

        return SessionRecord(
            session_number=0,
            session_date=datetime.utcnow().isoformat(),
            conversation_id=conversation_id,

            stage_of_change=dataset_row.get("stage_of_change", "unknown"),
            quit_attempt_history=dataset_row.get("quit_attempt_history", "unknown"),
            recommended_intervention=dataset_row.get("recommended_intervention", []),

            reasoning_summary=dataset_row.get("reasoning_summary"),
            confidence_score=dataset_row.get("confidence_score"),

            all_triggers=_normalize_signal_list(triggers),
            all_concerns=_normalize_signal_list(concerns),

            debate_rounds_used=dataset_row.get("debate_rounds_used"),
            vote_agreement=dataset_row.get("vote_agreement"),
            sanity_passed=dataset_row.get("sanity_passed", True),
            inconsistencies=dataset_row.get("inconsistencies", []),

            conversation_excerpt=conversation[:500] if conversation else None,
        )
