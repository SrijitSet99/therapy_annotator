"""
examples/test_pipeline_integration.py
---------------------------------------
Integration tests for the full adaptive pipeline graph.

Key design: ALL agent modules are imported at the top of this file
BEFORE any patching occurs. This ensures that patch() targets the
already-bound name in each module's namespace — the correct Python
mock pattern when using "from x import y" style imports.

No Ollama required.

Usage:
    python -m examples.test_pipeline_integration
    pytest examples/test_pipeline_integration.py -v
"""
import sys
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


# ── Pre-import ALL agent modules before any patching ─────────────────────
# This must happen at module level so patch() binds to the already-imported
# name rather than trying to intercept a future import.
import annotate.agents.extractors.advice_agent        # noqa: E402
import annotate.agents.extractors.attempt_agent       # noqa: E402
import annotate.agents.extractors.readiness_agent     # noqa: E402
import annotate.agents.extractors.concern_agent       # noqa: E402
import annotate.agents.extractors.trigger_agent       # noqa: E402
import annotate.agents.consensus.consensus_agent      # noqa: E402
import annotate.agents.reasoning.reasoning_agent      # noqa: E402
import annotate.agents.debate.debate_loop             # noqa: E402
import annotate.agents.confidence.confidence          # noqa: E402
import annotate.agents.revision.revision_agent        # noqa: E402
import annotate.agents.longitudinal.delta_agent       # noqa: E402
import annotate.graph.runner                          # noqa: E402
import annotate.graph.workflow                        # noqa: E402
import annotate.graph.parallel_workflow               # noqa: E402
import annotate.graph as graph                        # noqa: E402

SAMPLE_CHAT = (Path(__file__).parent.parent / "data/raw/sample_chat.txt").read_text()

# Patch targets — one per module that does "from annotate.llm.ollama_client import call_ollama"
_TARGETS = [
    "annotate.agents.extractors.advice_agent.call_ollama",
    "annotate.agents.extractors.attempt_agent.call_ollama",
    "annotate.agents.extractors.readiness_agent.call_ollama",
    "annotate.agents.extractors.concern_agent.call_ollama",
    "annotate.agents.extractors.trigger_agent.call_ollama",
    "annotate.agents.consensus.consensus_agent.call_ollama",
    "annotate.agents.reasoning.reasoning_agent.call_ollama",
    "annotate.agents.confidence.confidence.call_ollama",
    "annotate.agents.revision.revision_agent.call_ollama",
    "annotate.agents.longitudinal.delta_agent.call_ollama",
]


def _make_mock():
    """
    Context-aware mock: returns appropriate JSON based on which agent is calling.
    Detected by inspecting the prompt text for distinctive phrases.
    """
    def mock(prompt: str, **kwargs) -> str:
        p = prompt.lower()

        # Consensus voter
        if "voter" in p or ("unified" in p and "annotation" in p):
            return json.dumps({
                "stage_of_change": "preparation",
                "main_concern": "stress management",
                "primary_trigger": "work stress",
                "quit_attempt_history": "nicotine patch 6 weeks; cold turkey 2 days",
                "recommended_intervention": ["varenicline", "stress management plan"],
            })

        # Reasoning / final row
        if "reasoning_summary" in p or ("rationale" in p and "confidence_score" in p):
            return json.dumps({
                "stage_of_change": "preparation",
                "main_concern": "stress management",
                "primary_trigger": "work stress",
                "quit_attempt_history": "nicotine patch 6 weeks; cold turkey 2 days",
                "recommended_intervention": ["varenicline", "stress management plan"],
                "reasoning_summary": "Patient in preparation stage with prior attempts and clear quit plan.",
                "confidence_score": 0.87,
            })

        # Review agents
        if "reviewer" in p or ("review" in p and "inaccurac" in p):
            return json.dumps({"is_accurate": True, "issues": [], "suggestions": [], "severity": "none"})

        # Debate critic
        if "critic" in p or "agreement_reached" in p or ("hallucin" in p and "omission" in p):
            return json.dumps({"agreement_reached": True, "severity": "none", "issues": [], "suggestions": []})

        # Debate reviser
        if "reviser" in p or ("critic feedback" in p and "improved" in p):
            return json.dumps({"concerns": ["stress", "weight gain"], "primary_concern": "stress"})

        # Readiness extractor
        if "transtheoretical" in p or "readiness_stage" in p:
            return json.dumps({"readiness_stage": "preparation", "evidence": "patient set quit date"})

        # Trigger extractor
        if "smoking trigger" in p or ("situational" in p and "emotional" in p):
            return json.dumps({"triggers": ["work stress", "morning coffee"], "trigger_type": "mixed"})

        # Concern extractor
        if "barrier" in p or ("concern" in p and "fear" in p):
            return json.dumps({"concerns": ["stress", "weight gain"], "primary_concern": "stress"})

        # Attempt extractor
        if "quit attempt" in p or "prior attempt" in p:
            return json.dumps({"attempts": ["patch 6 weeks", "cold turkey 2 days"], "most_recent_attempt": "patch 6 weeks"})

        # Advice extractor
        if "cessation advice" in p or ("recommendation" in p and "counselor" in p):
            return json.dumps({"advice": ["varenicline", "stress management"], "evidence": ["counselor mentioned varenicline"]})

        # Generic fallback
        return json.dumps({"confidence": 0.95, "issues": []})

    return mock


def _all_mocked():
    """Return an ExitStack context that patches all ollama call sites."""
    mock_fn = _make_mock()
    stack = ExitStack()
    for target in _TARGETS:
        stack.enter_context(patch(target, mock_fn))
    return stack


def _reset_graph():
    """Force LangGraph to rebuild on next invocation."""
    graph.runner._graph = None


# ============================================================================
# Tests
# ============================================================================

def test_graph_builds_successfully():
    """LangGraph workflow compiles without errors."""
    g = graph.workflow.build_graph()
    assert g is not None
    print("PASS test_graph_builds_successfully")


def test_full_pipeline_end_to_end():
    """Full adaptive pipeline produces a valid DatasetRow with research metadata."""
    _reset_graph()
    with _all_mocked():
        result = graph.runner.run_pipeline(SAMPLE_CHAT, conversation_id="test_e2e")

    # Core clinical fields
    assert result.get("stage_of_change") == "preparation", f"got: {result.get('stage_of_change')}"
    assert isinstance(result.get("recommended_intervention"), list)
    assert len(result["recommended_intervention"]) > 0

    # Research metadata fields all present
    assert "debate_rounds_used" in result
    assert "vote_agreement"     in result
    assert "sanity_passed"      in result
    assert "inconsistencies"    in result
    assert isinstance(result["inconsistencies"], list)

    # Sanity should pass for a clean consistent annotation
    assert result["sanity_passed"] is True, f"sanity failed: {result['inconsistencies']}"

    print(f"PASS test_full_pipeline_end_to_end  "
          f"(stage={result['stage_of_change']}, "
          f"debate={result['debate_rounds_used']}, "
          f"agreement={result['vote_agreement']}, "
          f"sanity={result['sanity_passed']})")
    _reset_graph()


def test_parallel_workflow_end_to_end():
    """LangGraph-free parallel_workflow also produces a valid DatasetRow."""
    with _all_mocked():
        result = graph.parallel_workflow.run_parallel_pipeline(
            SAMPLE_CHAT, conversation_id="parallel_test"
        )
    assert result.get("stage_of_change") == "preparation"
    assert "sanity_passed" in result
    assert "vote_agreement" in result
    print(f"PASS test_parallel_workflow_end_to_end  (stage={result['stage_of_change']})")


def test_pipeline_persists_to_memory():
    """Pipeline stores the case in AgentMemory and it can be retrieved."""
    from annotate.agents.memory.agent_memory import AgentMemory
    _reset_graph()
    memory = AgentMemory()

    with _all_mocked():
        graph.runner.run_pipeline(SAMPLE_CHAT, conversation_id="mem_test", memory=memory)

    cases = memory.retrieve_similar_cases("preparation")
    assert len(cases) >= 1
    assert cases[0]["conversation_id"] == "mem_test"
    print("PASS test_pipeline_persists_to_memory")
    _reset_graph()


def test_sanity_fields_always_in_result():
    """sanity_passed and inconsistencies are always present even on minimal input."""
    _reset_graph()
    with _all_mocked():
        result = graph.runner.run_pipeline("Short chat.", conversation_id="sanity_test")

    assert "sanity_passed"   in result
    assert "inconsistencies" in result
    assert isinstance(result["sanity_passed"], bool)
    print(f"PASS test_sanity_fields_always_in_result  (sanity_passed={result['sanity_passed']})")
    _reset_graph()


def test_debate_skips_on_high_confidence():
    """Debate loop is skipped when the external confidence check meets threshold."""
    from annotate.agents.debate.debate_loop import maybe_debate
    output = {"concerns": ["stress"], "primary_concern": "stress"}
    # No mock needed — should short-circuit before any LLM call
    result = maybe_debate("concern", output, SAMPLE_CHAT, confidence_threshold=0.6, confidence_score=0.95)
    assert result["_debate_rounds"] == 0
    assert result["_debate_converged"] is True
    print("PASS test_debate_skips_on_high_confidence")


def test_debate_runs_and_converges_on_low_confidence():
    """Debate loop runs and terminates when critic approves."""
    from unittest.mock import MagicMock
    from annotate.agents.debate.debate_loop import maybe_debate
    output = {"concerns": ["stress"]}
    high_conf = MagicMock(confidence=0.8, issues=[])
    with patch("annotate.agents.debate.debate_loop.revise_signal", return_value=output), \
         patch("annotate.agents.debate.debate_loop.run_confidence_check", return_value=high_conf):
        result = maybe_debate("concern", output, SAMPLE_CHAT, confidence_threshold=0.6, confidence_score=0.3)
    assert result["_debate_rounds"] == 1
    assert result["_debate_converged"] is True
    print("PASS test_debate_runs_and_converges_on_low_confidence")


def test_debate_stops_when_revised_confidence_crosses_threshold():
    """Debate loop converges as soon as revised output clears the confidence threshold."""
    from unittest.mock import MagicMock
    from annotate.agents.debate.debate_loop import maybe_debate

    output = {"concerns": ["stress"]}
    revised = {"concerns": ["stress", "weight gain"]}

    high_conf = MagicMock(confidence=0.9, issues=[])
    with patch("annotate.agents.debate.debate_loop.revise_signal", return_value=revised), \
         patch("annotate.agents.debate.debate_loop.run_confidence_check", return_value=high_conf):
        result = maybe_debate("concern", output, SAMPLE_CHAT, confidence_threshold=0.6, confidence_score=0.2)

    assert result["_debate_rounds"] == 1
    assert result["_debate_converged"] is True
    print("PASS test_debate_stops_when_revised_confidence_crosses_threshold")


def test_agent_manager_disagreement_score():
    """Cross-signal disagreement is non-zero when confidences differ."""
    from annotate.agents.manager.agent_manager import AgentManager
    mgr = AgentManager()
    state = {
        "chat":      SAMPLE_CHAT,
        "confidence_results": {
            "advice": {"confidence": 0.9},
            "attempt": {"confidence": 0.3},
            "readiness": {"confidence": 0.8},
            "concern": {"confidence": 0.4},
            "triggers": {"confidence": 0.7},
        },
    }
    score = mgr.compute_cross_signal_disagreement(state)
    assert isinstance(score, float)
    assert score > 0.0, "Expected non-zero disagreement for varied confidences"
    print(f"PASS test_agent_manager_disagreement_score  (score={score:.4f})")


def test_registry_discovers_correct_agents():
    """Registry finds the extractor agents, including advice_agent."""
    import annotate.agents.registry as reg_mod
    reg_mod._default_registry = None   # force fresh discovery

    from annotate.agents.registry import get_registry
    registry = get_registry()

    assert "attempt_agent"   in registry.agent_names
    assert "readiness_agent" in registry.agent_names
    assert "concern_agent"   in registry.agent_names
    assert "trigger_agent"   in registry.agent_names
    assert "advice_agent"    in registry.agent_names

    reg_mod._default_registry = None
    print(f"PASS test_registry_discovers_correct_agents  ({registry.agent_names})")


def test_sanity_check_catches_stage_conflict():
    """Sanity check flags precontemplation + aggressive intervention."""
    from annotate.agents.judge.sanity_check_agent import sanity_check_agent
    state = {
        "chat": SAMPLE_CHAT,
        "consensus": {
            "stage_of_change": "precontemplation",
            "main_concern": "no motivation",
            "primary_trigger": "stress",
            "recommended_intervention": ["quit immediately", "set a quit date today"],
            "vote_agreement": 0.8,
        },
        "dataset_row": {"confidence_score": 0.5},
    }
    result = sanity_check_agent(state)
    assert result["sanity_check"]["passed"] is False
    assert len(result["sanity_check"]["inconsistencies"]) >= 1
    print("PASS test_sanity_check_catches_stage_conflict")


def test_multi_voter_consensus_majority():
    """Consensus agent selects majority stage when voters agree."""
    from annotate.agents.consensus.consensus_agent import _majority, _agreement_score
    values = ["preparation", "preparation", "contemplation"]
    assert _majority(values) == "preparation"
    assert abs(_agreement_score(values) - 2/3) < 0.01
    print("PASS test_multi_voter_consensus_majority")


def test_agent_memory_rolling_accuracy():
    """AgentMemory rolling window computes correct mean confidence."""
    from annotate.agents.memory.agent_memory import AgentMemory
    mem = AgentMemory()
    for conf in [0.8, 0.6, 0.7, 0.9]:
        mem.record_run("readiness_agent", confidence=conf)
    acc = mem.get_accuracy("readiness_agent")
    assert acc is not None
    assert abs(acc - 0.75) < 0.01   # mean of 0.8, 0.6, 0.7, 0.9
    print(f"PASS test_agent_memory_rolling_accuracy  (mean_conf={acc:.3f})")


def test_evaluator_full_metrics():
    """Evaluation framework produces expected scores on synthetic data."""
    from annotate.eval.evaluator import evaluate_against_gold
    preds = [
        {
            "stage_of_change": "preparation",
            "main_concern": "stress at work",
            "primary_trigger": "work stress",
            "recommended_intervention": ["varenicline", "NRT"],
            "confidence_score": 0.87,
            "debate_rounds_used": 1,
            "vote_agreement": 0.9,
            "sanity_passed": True,
        },
        {
            "stage_of_change": "precontemplation",
            "main_concern": "no motivation to quit",
            "primary_trigger": "stress relief",
            "recommended_intervention": ["motivational interviewing"],
            "confidence_score": 0.75,
            "debate_rounds_used": 0,
            "vote_agreement": 1.0,
            "sanity_passed": True,
        },
    ]
    golds = [
        {
            "stage_of_change": "preparation",
            "main_concern": "stress at work",
            "primary_trigger": "work stress",
            "recommended_intervention": ["varenicline", "counseling"],
        },
        {
            "stage_of_change": "precontemplation",
            "main_concern": "not motivated to quit",
            "primary_trigger": "stress relief after work",
            "recommended_intervention": ["motivational interviewing"],
        },
    ]
    m = evaluate_against_gold(preds, golds)
    assert m["stage_of_change_accuracy"] == 1.0
    assert m["sanity_pass_rate"]         == 1.0
    assert m["mean_debate_rounds"]       == 0.5
    assert abs(m["mean_vote_agreement"]  - 0.95) < 0.01
    assert 0.0 <= m["aggregate_score"]   <= 1.0
    print(f"PASS test_evaluator_full_metrics  "
          f"(stage_acc={m['stage_of_change_accuracy']}, aggregate={m['aggregate_score']})")


# ============================================================================

if __name__ == "__main__":
    tests = [
        test_graph_builds_successfully,
        test_full_pipeline_end_to_end,
        test_parallel_workflow_end_to_end,
        test_pipeline_persists_to_memory,
        test_sanity_fields_always_in_result,
        test_debate_skips_on_high_confidence,
        test_debate_runs_and_converges_on_low_confidence,
        test_debate_stops_when_revised_confidence_crosses_threshold,
        test_agent_manager_disagreement_score,
        test_registry_discovers_correct_agents,
        test_sanity_check_catches_stage_conflict,
        test_multi_voter_consensus_majority,
        test_agent_memory_rolling_accuracy,
        test_evaluator_full_metrics,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:
            print(f"FAIL {t.__name__}: {exc}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"{passed}/{len(tests)} integration tests passed.")
    print(f"{'='*50}")
    if passed < len(tests):
        sys.exit(1)
