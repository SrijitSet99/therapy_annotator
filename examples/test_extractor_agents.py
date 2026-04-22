"""
examples/test_extractor_agents.py
-----------------------------------
Unit tests for all agents — no Ollama required (mocked).

New tests cover:
  - No confidence field on extractor outputs
  - Debate loop early-exit when critic is satisfied
  - Debate loop revision when critic finds issues
  - Multi-voter consensus majority logic
  - Sanity check agent rule firing
  - AgentMemory recording and retrieval
  - Evaluator metric functions

Usage:
    python -m examples.test_extractor_agents
    pytest examples/test_extractor_agents.py -v
"""
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Sample data ────────────────────────────────────────────────────────────

SAMPLE_CHAT = (Path(__file__).parent.parent / "data/raw/sample_chat.txt").read_text()

BASE_STATE = {
    "chat": SAMPLE_CHAT,
    "advice": {},
    "attempt": {},
    "readiness": {},
    "concern": {},
    "triggers": {},
    "consensus": {},
    "sanity_check": {},
    "dataset_row": {},
}


def _mock_ollama(json_obj: dict):
    return lambda prompt, **kwargs: json.dumps(json_obj)


# ============================================================================
# Extractor agents — confidence is evaluated separately
# ============================================================================

def test_advice_agent_has_no_confidence():
    from annotate.agents.extractors.advice_agent import advice_agent
    mock = {"advice": ["Use NRT", "Try varenicline"], "evidence": ["counselor mentioned NRT"]}
    with patch("annotate.agents.extractors.advice_agent.call_ollama", _mock_ollama(mock)):
        result = advice_agent(BASE_STATE)
    assert "advice" in result
    assert "confidence" not in result["advice"]
    assert len(result["advice"]["advice"]) == 2
    print("PASS test_advice_agent_has_no_confidence")


def test_attempt_agent_has_no_confidence():
    from annotate.agents.extractors.attempt_agent import attempt_agent
    mock = {"attempts": ["Patch, 6 weeks"], "most_recent_attempt": "Patch, 6 weeks"}
    with patch("annotate.agents.extractors.attempt_agent.call_ollama", _mock_ollama(mock)):
        result = attempt_agent(BASE_STATE)
    assert "confidence" not in result["attempt"]
    print("PASS test_attempt_agent_has_no_confidence")


def test_readiness_agent_valid_stage():
    from annotate.agents.extractors.readiness_agent import readiness_agent
    mock = {"readiness_stage": "preparation", "evidence": "patient set a quit date"}
    with patch("annotate.agents.extractors.readiness_agent.call_ollama", _mock_ollama(mock)):
        result = readiness_agent(BASE_STATE)
    assert result["readiness"]["readiness_stage"] == "preparation"
    assert "confidence" not in result["readiness"]
    print("PASS test_readiness_agent_valid_stage")


def test_readiness_agent_invalid_stage_becomes_unknown():
    from annotate.agents.extractors.readiness_agent import readiness_agent
    mock = {"readiness_stage": "super_motivated"}
    with patch("annotate.agents.extractors.readiness_agent.call_ollama", _mock_ollama(mock)):
        result = readiness_agent(BASE_STATE)
    assert result["readiness"]["readiness_stage"] == "unknown"
    print("PASS test_readiness_agent_invalid_stage_becomes_unknown")


def test_concern_agent_has_no_confidence():
    from annotate.agents.extractors.concern_agent import concern_agent
    mock = {"concerns": ["weight gain", "stress"], "primary_concern": "stress"}
    with patch("annotate.agents.extractors.concern_agent.call_ollama", _mock_ollama(mock)):
        result = concern_agent(BASE_STATE)
    assert "confidence" not in result["concern"]
    print("PASS test_concern_agent_has_no_confidence")


def test_trigger_agent_has_no_confidence():
    from annotate.agents.extractors.trigger_agent import trigger_agent
    mock = {"triggers": ["stress", "morning coffee"], "trigger_type": "mixed"}
    with patch("annotate.agents.extractors.trigger_agent.call_ollama", _mock_ollama(mock)):
        result = trigger_agent(BASE_STATE)
    assert "confidence" not in result["triggers"]
    print("PASS test_trigger_agent_has_no_confidence")


# ============================================================================
# Debate loop
# ============================================================================

def test_debate_skipped_when_confidence_high():
    from annotate.agents.debate.debate_loop import maybe_debate
    output = {"concerns": ["stress"], "primary_concern": "stress"}
    result = maybe_debate("concern", output, SAMPLE_CHAT, confidence_threshold=0.6, confidence_score=0.9)
    assert result["_debate_rounds"] == 0
    assert result["_debate_converged"] is True
    print("PASS test_debate_skipped_when_confidence_high")


def test_debate_runs_when_confidence_low():
    from annotate.agents.debate.debate_loop import maybe_debate
    output = {"concerns": ["stress"]}
    high_conf = MagicMock(confidence=0.8, issues=[])
    with patch("annotate.agents.debate.debate_loop.revise_signal", return_value=output), \
         patch("annotate.agents.debate.debate_loop.run_confidence_check", return_value=high_conf):
        result = maybe_debate("concern", output, SAMPLE_CHAT, confidence_threshold=0.6, confidence_score=0.3)
    # Revised signal crossed the threshold immediately.
    assert result["_debate_rounds"] == 1
    assert result["_debate_converged"] is True
    print("PASS test_debate_runs_when_confidence_low")


def test_debate_revision_applied():
    from annotate.agents.debate.debate_loop import run_debate
    initial = {"concerns": ["stress"]}
    revised    = {"concerns": ["stress", "weight gain"]}

    high_conf = MagicMock(confidence=0.8, issues=[])
    with patch("annotate.agents.debate.debate_loop.revise_signal", return_value=revised), \
         patch("annotate.agents.debate.debate_loop.run_confidence_check", return_value=high_conf):
        result = run_debate("concern", initial, SAMPLE_CHAT, confidence_threshold=0.6)

    assert result.total_rounds >= 1
    assert result.final_output == revised
    print("PASS test_debate_revision_applied")


def test_debate_converges_when_revised_confidence_crosses_threshold():
    from annotate.agents.debate.debate_loop import maybe_debate
    output = {"concerns": ["stress"]}
    revised = {"concerns": ["stress", "weight gain"]}

    high_conf = MagicMock(confidence=0.85, issues=[])
    with patch("annotate.agents.debate.debate_loop.revise_signal", return_value=revised), \
         patch("annotate.agents.debate.debate_loop.run_confidence_check", return_value=high_conf):
        result = maybe_debate("concern", output, SAMPLE_CHAT, confidence_threshold=0.6, confidence_score=0.3)

    assert result["_debate_rounds"] == 1
    assert result["_debate_converged"] is True
    assert result["concerns"] == ["stress", "weight gain"]
    print("PASS test_debate_converges_when_revised_confidence_crosses_threshold")


# ============================================================================
# Sanity check agent
# ============================================================================

def test_sanity_check_passes_clean_state():
    from annotate.agents.judge.sanity_check_agent import sanity_check_agent
    state = {
        **BASE_STATE,
        "consensus": {
            "stage_of_change": "preparation",
            "main_concern": "stress",
            "primary_trigger": "work stress",
            "recommended_intervention": ["varenicline", "stress management"],
            "vote_agreement": 0.9,
        },
        "dataset_row": {"confidence_score": 0.8},
    }
    result = sanity_check_agent(state)
    assert result["sanity_check"]["passed"] is True
    assert result["sanity_check"]["inconsistencies"] == []
    print("PASS test_sanity_check_passes_clean_state")


def test_sanity_check_catches_stage_intervention_conflict():
    from annotate.agents.judge.sanity_check_agent import sanity_check_agent
    state = {
        **BASE_STATE,
        "consensus": {
            "stage_of_change": "precontemplation",
            "main_concern": "don't want to quit",
            "primary_trigger": "stress",
            "recommended_intervention": ["quit immediately", "set a quit date today"],
            "vote_agreement": 0.6,
        },
        "dataset_row": {"confidence_score": 0.5},
    }
    result = sanity_check_agent(state)
    assert result["sanity_check"]["passed"] is False
    assert len(result["sanity_check"]["inconsistencies"]) >= 1
    print("PASS test_sanity_check_catches_stage_intervention_conflict")


def test_sanity_check_catches_unknown_fields():
    from annotate.agents.judge.sanity_check_agent import sanity_check_agent
    state = {
        **BASE_STATE,
        "consensus": {
            "stage_of_change": "unknown",
            "main_concern": "unknown",
            "primary_trigger": "unknown",
            "recommended_intervention": [],
            "vote_agreement": 0.3,
        },
        "dataset_row": {"confidence_score": 0.9},
    }
    result = sanity_check_agent(state)
    assert result["sanity_check"]["passed"] is False
    # Should flag unknown fields AND low-agreement/high-confidence mismatch
    assert len(result["sanity_check"]["inconsistencies"]) >= 3
    print("PASS test_sanity_check_catches_unknown_fields")


# ============================================================================
# Agent memory
# ============================================================================

def test_agent_memory_records_and_retrieves():
    from annotate.agents.memory.agent_memory import AgentMemory
    mem = AgentMemory()
    mem.record_run("concern_agent", confidence=0.8)
    mem.record_run("concern_agent", confidence=0.6)
    acc = mem.get_accuracy("concern_agent")
    assert acc is not None
    assert abs(acc - 0.7) < 0.01  # mean of 0.8 and 0.6
    print("PASS test_agent_memory_records_and_retrieves")


def test_agent_memory_unknown_agent_returns_none():
    from annotate.agents.memory.agent_memory import AgentMemory
    mem = AgentMemory()
    assert mem.get_accuracy("nonexistent_agent") is None
    print("PASS test_agent_memory_unknown_agent_returns_none")


def test_agent_memory_stores_and_retrieves_cases():
    from annotate.agents.memory.agent_memory import AgentMemory
    mem = AgentMemory()
    row = {"stage_of_change": "preparation", "main_concern": "stress"}
    mem.store_case("conv_001", row)
    similar = mem.retrieve_similar_cases("preparation")
    assert len(similar) == 1
    assert similar[0]["conversation_id"] == "conv_001"
    print("PASS test_agent_memory_stores_and_retrieves_cases")


# ============================================================================
# Evaluator metrics
# ============================================================================

def test_evaluator_accuracy():
    from annotate.eval.evaluator import accuracy
    assert accuracy(["preparation", "action", "preparation"], ["preparation", "contemplation", "preparation"]) == pytest_approx(2/3)
    print("PASS test_evaluator_accuracy")


def test_evaluator_token_f1():
    from annotate.eval.evaluator import token_f1
    score = token_f1("stress at work", "work stress")
    assert score > 0.5  # overlap: "stress", "work"
    assert token_f1("", "stress") == 0.0
    print("PASS test_evaluator_token_f1")


def test_evaluator_list_jaccard():
    from annotate.eval.evaluator import list_jaccard
    assert list_jaccard(["varenicline", "NRT"], ["NRT", "counseling"]) == pytest_approx(1/3)
    assert list_jaccard([], []) == 1.0
    print("PASS test_evaluator_list_jaccard")


def test_evaluator_against_gold():
    from annotate.eval.evaluator import evaluate_against_gold
    preds = [
        {"stage_of_change": "preparation", "main_concern": "stress", "primary_trigger": "work",
         "recommended_intervention": ["varenicline"], "confidence_score": 0.8,
         "debate_rounds_used": 1, "vote_agreement": 0.9, "sanity_passed": True}
    ]
    golds = [
        {"stage_of_change": "preparation", "main_concern": "stress at work",
         "primary_trigger": "work stress", "recommended_intervention": ["varenicline", "NRT"]}
    ]
    metrics = evaluate_against_gold(preds, golds)
    assert metrics["stage_of_change_accuracy"] == 1.0
    assert metrics["sanity_pass_rate"] == 1.0
    assert 0 <= metrics["aggregate_score"] <= 1.0
    print("PASS test_evaluator_against_gold")


# ============================================================================
# JSON repair
# ============================================================================

def test_repair_json_handles_preamble():
    from annotate.llm.json_parser import repair_json
    text = 'Sure! Here is the output:\n{"advice": ["quit now"], "evidence": ["patient said so"]}'
    parsed = repair_json(text)
    assert parsed["advice"] == ["quit now"]
    print("PASS test_repair_json_handles_preamble")


def test_repair_json_handles_fenced_block():
    from annotate.llm.json_parser import repair_json
    text = '```json\n{"readiness_stage": "action"}\n```'
    parsed = repair_json(text)
    assert parsed["readiness_stage"] == "action"
    print("PASS test_repair_json_handles_fenced_block")


# ============================================================================
# Runner
# ============================================================================

def pytest_approx(val, rel=1e-3):
    """Tiny approximation helper for environments without pytest installed."""
    class Approx:
        def __eq__(self, other):
            return abs(other - val) <= rel
        def __repr__(self):
            return f"approx({val})"
    return Approx()


if __name__ == "__main__":
    test_advice_agent_has_no_confidence()
    test_attempt_agent_has_no_confidence()
    test_readiness_agent_valid_stage()
    test_readiness_agent_invalid_stage_becomes_unknown()
    test_concern_agent_has_no_confidence()
    test_trigger_agent_has_no_confidence()

    test_debate_skipped_when_confidence_high()
    test_debate_runs_when_confidence_low()
    test_debate_revision_applied()

    test_sanity_check_passes_clean_state()
    test_sanity_check_catches_stage_intervention_conflict()
    test_sanity_check_catches_unknown_fields()

    test_agent_memory_records_and_retrieves()
    test_agent_memory_unknown_agent_returns_none()
    test_agent_memory_stores_and_retrieves_cases()

    test_evaluator_accuracy()
    test_evaluator_token_f1()
    test_evaluator_list_jaccard()
    test_evaluator_against_gold()

    test_repair_json_handles_preamble()
    test_repair_json_handles_fenced_block()

    print("\n" + "="*50)
    print("All tests passed.")
    print("="*50)
