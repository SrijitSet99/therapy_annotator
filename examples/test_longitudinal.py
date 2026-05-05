"""
examples/test_longitudinal.py
-------------------------------
Unit and integration tests for the longitudinal multi-session system.
No Ollama required.

Tests cover:
  - PatientMemory: create, load, save, append, recompute
  - stage_direction logic (all transition types)
  - context_summary generation (history injection)
  - delta_agent: structural diff computation
  - longitudinal_sanity_agent: all 7 rules
  - profile_update_agent: full profile lifecycle
  - Full multi-session pipeline (4 sessions, mocked LLM)

Usage:
    python -m examples.test_longitudinal
    pytest examples/test_longitudinal.py -v
"""
import sys
import json
import tempfile
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch


# Pre-import everything before patching
import annotate.agents.extractors.advice_agent
import annotate.agents.extractors.attempt_agent
import annotate.agents.extractors.readiness_agent
import annotate.agents.extractors.concern_agent
import annotate.agents.extractors.trigger_agent
import annotate.agents.consensus.consensus_agent
import annotate.agents.reasoning.reasoning_agent
import annotate.agents.longitudinal.delta_agent
import annotate.graph.runner
import annotate.graph.workflow

from annotate.schemas.longitudinal_schema import (
    SessionRecord, PatientProfile, SessionDelta,
    stage_direction, LongitudinalAnnotation,
)
from annotate.agents.memory.patient_memory import PatientMemory
from annotate.agents.longitudinal.context_agent import longitudinal_context_agent
from annotate.agents.longitudinal.delta_agent import delta_agent, _normalise_set
from annotate.agents.longitudinal.longitudinal_sanity import longitudinal_sanity_agent
from annotate.agents.longitudinal.profile_update_agent import profile_update_agent

SAMPLE_CHAT = (Path(__file__).parent.parent / "data/raw/sample_chat.txt").read_text()

_TARGETS = [
    "annotate.agents.extractors.advice_agent.call_ollama",
    "annotate.agents.extractors.attempt_agent.call_ollama",
    "annotate.agents.extractors.readiness_agent.call_ollama",
    "annotate.agents.extractors.concern_agent.call_ollama",
    "annotate.agents.extractors.trigger_agent.call_ollama",
    "annotate.agents.consensus.consensus_agent.call_ollama",
    "annotate.agents.reasoning.reasoning_agent.call_ollama",
    "annotate.agents.longitudinal.delta_agent.call_ollama",
]


def _make_mock(stage="preparation"):
    def mock(prompt: str, **kw) -> str:
        p = prompt.lower()
        if "voter" in p or ("unified" in p and "annotation" in p):
            return json.dumps({"stage_of_change": stage, "all_concerns": ["stress"],
                "all_triggers": ["work stress"], "quit_attempt_history": "patch 6 weeks",
                "recommended_intervention": ["varenicline", "stress management"], "confidence": 0.87})
        if "reasoning_summary" in p or ("rationale" in p and "confidence_score" in p):
            return json.dumps({"stage_of_change": stage, "all_concerns": ["stress"],
                "all_triggers": ["work stress"], "quit_attempt_history": "patch 6 weeks",
                "recommended_intervention": ["varenicline"], "reasoning_summary": "Patient progressing.",
                "confidence_score": 0.87})
        if "reviewer" in p or ("review" in p and "inaccurac" in p):
            return json.dumps({"is_accurate": True, "issues": [], "suggestions": [], "severity": "none"})
        if "critic" in p or "agreement_reached" in p:
            return json.dumps({"agreement_reached": True, "severity": "none", "issues": []})
        if "clinical_summary" in p or "delta" in p.lower():
            return json.dumps({"clinical_summary": "Patient progressed well.", "delta_confidence": 0.8})
        if "transtheoretical" in p or "readiness_stage" in p:
            return json.dumps({"readiness_stage": stage, "confidence": 0.82})
        if "smoking trigger" in p or ("situational" in p and "emotional" in p):
            return json.dumps({"triggers": ["work stress", "morning coffee"], "trigger_type": "mixed", "confidence": 0.8})
        if "barrier" in p or ("concern" in p and "fear" in p):
            return json.dumps({"concerns": ["stress", "weight gain"], "primary_concern": "stress", "confidence": 0.78})
        if "quit attempt" in p or "prior attempt" in p:
            return json.dumps({"attempts": ["patch 6 weeks"], "most_recent_attempt": "patch 6 weeks", "confidence": 0.85})
        if "cessation advice" in p or "advice" in p:
            return json.dumps({"advice": ["varenicline"], "evidence": ["counselor mentioned varenicline"], "confidence": 0.88})
        return json.dumps({"confidence": 0.75})
    return mock


def _all_mocked(stage="preparation"):
    fn = _make_mock(stage)
    stack = ExitStack()
    for t in _TARGETS:
        stack.enter_context(patch(t, fn))
    return stack


def _make_session(number: int, stage: str, triggers=None, concerns=None, interventions=None) -> SessionRecord:
    return SessionRecord(
        session_number=number,
        stage_of_change=stage,
        quit_attempt_history="patch 6 weeks",
        recommended_intervention=interventions or ["varenicline"],
        all_triggers=triggers or ["work stress"],
        all_concerns=concerns or ["stress"],
        confidence_score=0.85,
    )


# ============================================================================
# 1. stage_direction logic
# ============================================================================

def test_stage_direction_progress():
    assert stage_direction("precontemplation", "contemplation") == "progress"
    assert stage_direction("contemplation", "preparation") == "progress"
    assert stage_direction("preparation", "action") == "progress"
    print("PASS test_stage_direction_progress")


def test_stage_direction_regression():
    assert stage_direction("action", "preparation") == "regression"
    assert stage_direction("preparation", "contemplation") == "regression"
    print("PASS test_stage_direction_regression")


def test_stage_direction_relapse():
    assert stage_direction("action", "relapse") == "relapse"
    assert stage_direction("maintenance", "relapse") == "relapse"
    print("PASS test_stage_direction_relapse")


def test_stage_direction_recovery():
    assert stage_direction("relapse", "contemplation") == "recovery"
    assert stage_direction("relapse", "preparation") == "recovery"
    print("PASS test_stage_direction_recovery")


def test_stage_direction_stable():
    assert stage_direction("contemplation", "contemplation") == "stable"
    assert stage_direction("action", "action") == "stable"
    print("PASS test_stage_direction_stable")


def test_stage_direction_unknown():
    assert stage_direction("unknown", "preparation") == "unknown"
    assert stage_direction("preparation", "unknown") == "unknown"
    print("PASS test_stage_direction_unknown")


# ============================================================================
# 2. PatientMemory: create, append, recompute, persist
# ============================================================================

def test_patient_memory_creates_new_profile():
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PatientMemory(patients_dir=tmpdir)
        profile = pm.load_or_create("test_patient")
        assert profile.patient_id == "test_patient"
        assert profile.session_count == 0
        assert profile.is_first_session
    print("PASS test_patient_memory_creates_new_profile")


def test_patient_memory_append_and_recompute():
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PatientMemory(patients_dir=tmpdir)
        profile = pm.load_or_create("p001")

        # Session 1
        s1 = _make_session(0, "precontemplation", triggers=["work stress"], concerns=["no motivation"])
        profile = pm.append_session(profile, s1)
        assert profile.session_count == 1
        assert profile.stage_trajectory == ["precontemplation"]
        assert profile.current_stage == "precontemplation"
        assert "work stress" in profile.all_triggers_ever

        # Session 2
        s2 = _make_session(0, "contemplation", triggers=["work stress", "morning coffee"], concerns=["stress"])
        profile = pm.append_session(profile, s2)
        assert profile.session_count == 2
        assert profile.stage_trajectory == ["precontemplation", "contemplation"]
        assert profile.stage_direction_history == ["progress"]
        assert "work stress" in profile.persistent_triggers  # in both sessions
        assert "morning coffee" not in profile.persistent_triggers  # only in session 2

    print("PASS test_patient_memory_append_and_recompute")


def test_patient_memory_stagnation_detection():
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PatientMemory(patients_dir=tmpdir)
        profile = pm.load_or_create("p_stag")
        for _ in range(4):
            s = _make_session(0, "contemplation")
            profile = pm.append_session(profile, s)
        assert profile.stagnation_count == 3  # 3 consecutive stable transitions
    print("PASS test_patient_memory_stagnation_detection")


def test_patient_memory_relapse_counting():
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PatientMemory(patients_dir=tmpdir)
        profile = pm.load_or_create("p_relapse")
        for stage in ["preparation", "action", "relapse", "contemplation", "preparation"]:
            s = _make_session(0, stage)
            profile = pm.append_session(profile, s)
        assert profile.relapse_count == 1
        assert "relapse" in profile.stage_trajectory
    print("PASS test_patient_memory_relapse_counting")


def test_patient_memory_best_stage():
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PatientMemory(patients_dir=tmpdir)
        profile = pm.load_or_create("p_best")
        for stage in ["precontemplation", "contemplation", "action", "relapse", "contemplation"]:
            s = _make_session(0, stage)
            profile = pm.append_session(profile, s)
        assert profile.best_stage_reached == "action"
    print("PASS test_patient_memory_best_stage")


def test_patient_memory_save_and_reload():
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PatientMemory(patients_dir=tmpdir)
        profile = pm.load_or_create("p_persist")
        s1 = _make_session(0, "preparation", triggers=["stress"])
        profile = pm.append_session(profile, s1)
        pm.save(profile)

        # Reload in a new instance
        pm2 = PatientMemory(patients_dir=tmpdir)
        reloaded = pm2.load_or_create("p_persist")
        assert reloaded.session_count == 1
        assert reloaded.stage_trajectory == ["preparation"]
        assert "stress" in reloaded.all_triggers_ever
    print("PASS test_patient_memory_save_and_reload")


def test_context_summary_is_structured():
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PatientMemory(patients_dir=tmpdir)
        profile = pm.load_or_create("p_ctx")
        for stage in ["precontemplation", "contemplation"]:
            s = _make_session(0, stage, triggers=["work stress"], concerns=["stress"])
            profile = pm.append_session(profile, s)
        assert "Trajectory:" in profile.context_summary
        assert "precontemplation → contemplation" in profile.context_summary
        assert "Current stage: contemplation" in profile.context_summary
    print("PASS test_context_summary_is_structured")


# ============================================================================
# 3. longitudinal_context_agent
# ============================================================================

def test_context_agent_noop_first_session():
    state = {"chat": "Hello", "patient_profile": None}
    result = longitudinal_context_agent(state)
    assert result["enriched_chat"] == "Hello"
    assert result["longitudinal_flags"] == []
    print("PASS test_context_agent_noop_first_session")


def test_context_agent_injects_history():
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PatientMemory(patients_dir=tmpdir)
        profile = pm.load_or_create("p_inject")
        s1 = _make_session(0, "precontemplation", triggers=["stress"])
        profile = pm.append_session(profile, s1)

        state = {"chat": "Current session text", "patient_profile": profile}
        result = longitudinal_context_agent(state)
        assert "PATIENT HISTORY" in result["enriched_chat"]
        assert "precontemplation" in result["enriched_chat"]
        assert "Current session text" in result["enriched_chat"]
    print("PASS test_context_agent_injects_history")


def test_context_agent_raises_stagnation_flag():
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PatientMemory(patients_dir=tmpdir)
        profile = pm.load_or_create("p_stag2")
        for _ in range(4):
            s = _make_session(0, "contemplation")
            profile = pm.append_session(profile, s)

        state = {"chat": "...", "patient_profile": profile}
        result = longitudinal_context_agent(state)
        assert any("STAGNATION" in f for f in result["longitudinal_flags"])
    print("PASS test_context_agent_raises_stagnation_flag")


# ============================================================================
# 4. delta_agent
# ============================================================================

def test_delta_agent_noop_first_session():
    state = {"patient_profile": None, "dataset_row": {}}
    result = delta_agent(state)
    assert result["session_delta"] is None
    print("PASS test_delta_agent_noop_first_session")


def test_delta_agent_computes_progress():
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PatientMemory(patients_dir=tmpdir)
        profile = pm.load_or_create("p_delta")
        s1 = _make_session(0, "contemplation", triggers=["stress"], concerns=["motivation"])
        profile = pm.append_session(profile, s1)

        state = {
            "patient_profile": profile,
            "dataset_row": {"stage_of_change": "preparation",
                           "all_concerns": ["stress"], "all_triggers": ["work"],
                           "recommended_intervention": ["varenicline"]},
            "triggers": {"triggers": ["work stress", "morning coffee"]},
            "concern":  {"concerns": ["stress", "weight gain"]},
            "chat": "patient set a quit date",
        }

        delta_mock = json.dumps({"clinical_summary": "Patient moved to preparation.", "delta_confidence": 0.8})
        with patch("annotate.agents.longitudinal.delta_agent.call_ollama", lambda p, **kw: delta_mock):
            result = delta_agent(state)

        d = result["session_delta"]
        assert d["stage_before"] == "contemplation"
        assert d["stage_after"]  == "preparation"
        assert d["stage_direction"] == "progress"
        assert d["stage_changed"] is True
        assert isinstance(d["new_triggers"], list)
        assert isinstance(d["new_concerns"], list)
        assert d["clinical_summary"] == "Patient moved to preparation."
    print("PASS test_delta_agent_computes_progress")


def test_delta_agent_detects_relapse():
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PatientMemory(patients_dir=tmpdir)
        profile = pm.load_or_create("p_relapse2")
        s1 = _make_session(0, "action")
        profile = pm.append_session(profile, s1)

        state = {
            "patient_profile": profile,
            "dataset_row": {"stage_of_change": "relapse", "all_concerns": ["stress"],
                           "all_triggers": ["work"], "recommended_intervention": []},
            "triggers": {"triggers": ["stress"]},
            "concern":  {"concerns": ["stress"]},
            "chat": "I started smoking again last week",
        }
        delta_mock = json.dumps({"clinical_summary": "Patient relapsed.", "delta_confidence": 0.7})
        with patch("annotate.agents.longitudinal.delta_agent.call_ollama", lambda p, **kw: delta_mock):
            result = delta_agent(state)

        d = result["session_delta"]
        assert d["relapse_detected"] is True
        assert d["stage_direction"] == "relapse"
    print("PASS test_delta_agent_detects_relapse")


# ============================================================================
# 5. longitudinal_sanity_agent
# ============================================================================

def test_longitudinal_sanity_passes_clean_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PatientMemory(patients_dir=tmpdir)
        profile = pm.load_or_create("p_lsanity")
        s1 = _make_session(0, "contemplation", triggers=["stress"])
        profile = pm.append_session(profile, s1)

        state = {
            "patient_profile": profile,
            "session_delta": {"stage_before": "contemplation", "stage_after": "preparation",
                             "stage_direction": "progress", "relapse_detected": False,
                             "new_interventions": [], "stagnation_detected": False},
            "dataset_row": {"stage_of_change": "preparation", "all_concerns": ["stress"],
                           "recommended_intervention": ["varenicline"]},
            "chat": "I'm ready to set a quit date",
        }
        result = longitudinal_sanity_agent(state)
        assert result["longitudinal_sanity"]["passed"] is True
    print("PASS test_longitudinal_sanity_passes_clean_state")


def test_longitudinal_sanity_catches_stage_skip():
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PatientMemory(patients_dir=tmpdir)
        profile = pm.load_or_create("p_skip")
        s1 = _make_session(0, "precontemplation")
        profile = pm.append_session(profile, s1)

        state = {
            "patient_profile": profile,
            "session_delta": {"stage_before": "precontemplation", "stage_after": "action",
                             "stage_direction": "progress", "relapse_detected": False,
                             "new_interventions": [], "stagnation_detected": False},
            "dataset_row": {"stage_of_change": "action", "all_concerns": ["stress"],
                           "recommended_intervention": []},
            "chat": "I've been smoke-free",
        }
        result = longitudinal_sanity_agent(state)
        assert result["longitudinal_sanity"]["passed"] is False
        assert any("STAGE_SKIP" in f for f in result["longitudinal_sanity"]["flags"])
    print("PASS test_longitudinal_sanity_catches_stage_skip")


def test_longitudinal_sanity_catches_silent_relapse():
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PatientMemory(patients_dir=tmpdir)
        profile = pm.load_or_create("p_silent")
        s1 = _make_session(0, "action")
        profile = pm.append_session(profile, s1)

        state = {
            "patient_profile": profile,
            "session_delta": {"stage_before": "action", "stage_after": "relapse",
                             "stage_direction": "relapse", "relapse_detected": True,
                             "new_interventions": [], "stagnation_detected": False},
            "dataset_row": {"stage_of_change": "relapse", "all_concerns": ["stress"],
                           "recommended_intervention": []},
            "chat": "things have been stressful but I'm managing",  # no relapse language
        }
        result = longitudinal_sanity_agent(state)
        assert result["longitudinal_sanity"]["passed"] is False
        assert any("SILENT_RELAPSE" in f for f in result["longitudinal_sanity"]["flags"])
    print("PASS test_longitudinal_sanity_catches_silent_relapse")


def test_longitudinal_sanity_catches_stagnation():
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PatientMemory(patients_dir=tmpdir)
        profile = pm.load_or_create("p_stag3")
        for _ in range(5):
            s = _make_session(0, "contemplation")
            profile = pm.append_session(profile, s)

        state = {
            "patient_profile": profile,
            "session_delta": {"stage_before": "contemplation", "stage_after": "contemplation",
                             "stage_direction": "stable", "relapse_detected": False,
                             "new_interventions": [], "stagnation_detected": True},
            "dataset_row": {"stage_of_change": "contemplation", "all_concerns": ["not ready"],
                           "recommended_intervention": []},
            "chat": "still thinking about it",
        }
        result = longitudinal_sanity_agent(state)
        assert any("STAGNATION" in f for f in result["longitudinal_sanity"]["flags"])
    print("PASS test_longitudinal_sanity_catches_stagnation")


# ============================================================================
# 6. Full multi-session pipeline (4 sessions, mocked LLM)
# ============================================================================

def test_multisession_pipeline_four_sessions():
    """
    Simulate 4 sessions of a patient journey:
    precontemplation → contemplation → preparation → action
    Verify trajectory, delta, profile persistence.
    """
    stages = ["precontemplation", "contemplation", "preparation", "action"]
    conversations = [
        "Patient says they are not ready to quit.",
        "Patient says they are thinking about quitting.",
        "Patient sets a quit date for two weeks.",
        "Patient has been smoke-free for two weeks.",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PatientMemory(patients_dir=tmpdir)
        graph.runner._graph = None

        results = []
        for i, (conv, stage) in enumerate(zip(conversations, stages), 1):
            with _all_mocked(stage=stage):
                from annotate.graph.runner import run_session as rs
                # Patch the patient memory used by the workflow node
                with patch("annotate.graph.workflow._patient_memory", pm):
                    result = rs(conv, patient_id="p_test_4",
                                conversation_id=f"s{i:03d}", patient_memory=pm)
            results.append(result)
            graph.runner._graph = None

        # Verify all 4 sessions completed
        assert len(results) == 4

        # Verify trajectory
        profile = pm.load_or_create("p_test_4")
        assert profile.session_count == 4
        assert profile.stage_trajectory == stages
        assert profile.best_stage_reached == "action"
        assert profile.relapse_count == 0

        # Verify deltas exist from session 2 onward
        for i, result in enumerate(results[1:], 2):
            lo = result.get("longitudinal_output", {})
            delta = lo.get("delta")
            assert delta is not None, f"Session {i} should have a delta"
            assert delta.get("from_session") == i - 1
            assert delta.get("to_session") == i

        # Verify direction history
        assert profile.stage_direction_history == ["progress", "progress", "progress"]

        print(f"PASS test_multisession_pipeline_four_sessions  "
              f"(trajectory={' → '.join(profile.stage_trajectory)})")

        graph.runner._graph = None


# ============================================================================

if __name__ == "__main__":
    tests = [
        test_stage_direction_progress,
        test_stage_direction_regression,
        test_stage_direction_relapse,
        test_stage_direction_recovery,
        test_stage_direction_stable,
        test_stage_direction_unknown,
        test_patient_memory_creates_new_profile,
        test_patient_memory_append_and_recompute,
        test_patient_memory_stagnation_detection,
        test_patient_memory_relapse_counting,
        test_patient_memory_best_stage,
        test_patient_memory_save_and_reload,
        test_context_summary_is_structured,
        test_context_agent_noop_first_session,
        test_context_agent_injects_history,
        test_context_agent_raises_stagnation_flag,
        test_delta_agent_noop_first_session,
        test_delta_agent_computes_progress,
        test_delta_agent_detects_relapse,
        test_longitudinal_sanity_passes_clean_state,
        test_longitudinal_sanity_catches_stage_skip,
        test_longitudinal_sanity_catches_silent_relapse,
        test_longitudinal_sanity_catches_stagnation,
        test_multisession_pipeline_four_sessions,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:
            print(f"FAIL {t.__name__}: {exc}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*55}")
    print(f"{passed}/{len(tests)} longitudinal tests passed.")
    print(f"{'='*55}")
    if passed < len(tests):
        sys.exit(1)
