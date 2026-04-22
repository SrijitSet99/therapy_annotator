"""
graph/workflow.py
------------------
Adaptive LangGraph workflow with full longitudinal coherence support.

Topology
--------
                    longitudinal_context        ← inject history (no-op session 1)
                           │
                    parallel_extract            ← AgentManager fanout
                           │
                    confidence_check            ← evaluate all signal confidences
                           │
              ┌────────────┴────────────┐
         (< threshold)              (>= threshold)
              │                          │
              │                    multi_consensus   ← 3 voters + arbitration
           revision                     │
              │                      reasoning      ← final DatasetRow
              └──────→ confidence_check    │
                         (loop back)   delta_compute
                                          │
                                   longitudinal_sanity
                                          │
                                   profile_update     ← persist PatientProfile
                                          │
                                         END
"""
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from annotate.graph.state import GraphState
from annotate.agents.manager.agent_manager import AgentManager
from annotate.agents.consensus.consensus_agent import consensus_agent
from annotate.agents.judge.sanity_check_agent import sanity_check_agent
from annotate.agents.reasoning.reasoning_agent import reasoning_agent
from annotate.agents.longitudinal.context_agent import longitudinal_context_agent
from annotate.agents.longitudinal.delta_agent import delta_agent
from annotate.agents.longitudinal.longitudinal_sanity import longitudinal_sanity_agent
from annotate.agents.longitudinal.profile_update_agent import profile_update_agent
from annotate.agents.memory.patient_memory import PatientMemory
from annotate.config.settings import CONFIDENCE_DEBATE_THRESHOLD, MAX_REVISION_ROUNDS
from annotate.utils.logger import log

_manager = AgentManager()
_patient_memory = PatientMemory()


# ---------------------------------------------------------------------------
# Node wrappers
# ---------------------------------------------------------------------------

def longitudinal_context_node(state: GraphState) -> Dict[str, Any]:
    log("workflow: [longitudinal_context] ...")
    return longitudinal_context_agent(state)


def parallel_extract_node(state: GraphState, enable_debate: bool = False) -> Dict[str, Any]:
    log("workflow: [parallel_extract] ...")
    effective_state = dict(state)
    if state.get("enriched_chat"):
        effective_state["chat"] = state["enriched_chat"]
    outputs = _manager.run_extractors(effective_state, enable_debate=enable_debate)
    return outputs


def confidence_check_node(state: GraphState) -> Dict[str, Any]:
    """
    Runs per-signal confidence agents and aggregates results into
    confidence_results. Also resets revision_count to 0 on the very
    first call (i.e. when it hasn't been set yet).
    """
    log("workflow: [confidence_check] ...")
    results = _manager.run_confidence_check(state)
    # Initialise revision_count if this is the first confidence check
    if state.get("revision_count") is None:
        results["revision_count"] = 0
    return results


def revision_node(state: GraphState) -> Dict[str, Any]:
    """
    Runs per-signal revision agents for all signals that are below
    threshold, then increments revision_count.
    """
    log("workflow: [revision] ...")
    revised = _manager.run_revision(state)
    revised["revision_count"] = state.get("revision_count", 0) + 1
    return revised


def consensus_node(state: GraphState) -> Dict[str, Any]:
    log("workflow: [multi_consensus] ...")
    return consensus_agent(state)


def sanity_check_node(state: GraphState) -> Dict[str, Any]:
    log("workflow: [sanity_check] ...")
    return sanity_check_agent(state)


def reasoning_node(state: GraphState) -> Dict[str, Any]:
    log("workflow: [reasoning] ...")
    return reasoning_agent(state)


def delta_compute_node(state: GraphState) -> Dict[str, Any]:
    log("workflow: [delta_compute] ...")
    return delta_agent(state)


def longitudinal_sanity_node(state: GraphState) -> Dict[str, Any]:
    log("workflow: [longitudinal_sanity] ...")
    return longitudinal_sanity_agent(state)


def profile_update_node(state: GraphState) -> Dict[str, Any]:
    log("workflow: [profile_update] ...")
    return profile_update_agent(state, _patient_memory)


# ---------------------------------------------------------------------------
# Confidence router — drives the confidence→revision loop
# ---------------------------------------------------------------------------

def confidence_router(state: GraphState) -> str:
    """
    Route from confidence_check:
      - If ALL signals meet the threshold  → 'multi_consensus'
      - Otherwise                          → 'revision'
    """
    revision_count = state.get("revision_count", 0)
    confidence_results = state.get("confidence_results", {})

    if revision_count >= MAX_REVISION_ROUNDS:
        log(
            f"workflow: confidence_router → max revision rounds reached "
            f"({MAX_REVISION_ROUNDS}); proceeding to consensus."
        )
        return "multi_consensus"

    # Check every signal; route to revision if any are below threshold
    signals = ("advice", "attempt", "readiness", "concern", "triggers")
    for sig in signals:
        conf = confidence_results.get(sig, {}).get("confidence", 0.0)
        if conf < CONFIDENCE_DEBATE_THRESHOLD:
            log(f"workflow: confidence_router → '{sig}' confidence={conf:.3f} "
                f"< {CONFIDENCE_DEBATE_THRESHOLD} (revision {revision_count + 1})")
            return "revision"

    log("workflow: confidence_router → all signals above threshold, proceeding to consensus.")
    return "multi_consensus"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(enable_debate: bool = False):
    workflow = StateGraph(GraphState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    workflow.add_node("longitudinal_context",     longitudinal_context_node)
    workflow.add_node(
        "parallel_extract",
        lambda state: parallel_extract_node(state, enable_debate=enable_debate),
    )
    workflow.add_node("confidence_check",         confidence_check_node)
    workflow.add_node("revision",                 revision_node)
    workflow.add_node("multi_consensus",          consensus_node)
    workflow.add_node("reasoning",                reasoning_node)
    workflow.add_node("delta_compute",            delta_compute_node)
    workflow.add_node("longitudinal_sanity_check", longitudinal_sanity_node)
    workflow.add_node("profile_update",           profile_update_node)

    # ── Entry ──────────────────────────────────────────────────────────────
    workflow.set_entry_point("longitudinal_context")

    # ── Linear prefix ──────────────────────────────────────────────────────
    workflow.add_edge("longitudinal_context", "parallel_extract")
    workflow.add_edge("parallel_extract",     "confidence_check")

    # ── Confidence → revision loop ─────────────────────────────────────────
    # confidence_check branches: low-confidence → revision, else → multi_consensus
    workflow.add_conditional_edges(
        "confidence_check",
        confidence_router,
        {
            "revision":      "revision",
            "multi_consensus": "multi_consensus",
        },
    )
    # After revision, re-evaluate confidence (closing the loop)
    workflow.add_edge("revision", "confidence_check")

    # ── Synthesis chain ────────────────────────────────────────────────────
    workflow.add_edge("multi_consensus", "reasoning")

    # ── Longitudinal chain ─────────────────────────────────────────────────
    workflow.add_edge("reasoning",                "delta_compute")
    workflow.add_edge("delta_compute",            "longitudinal_sanity_check")
    workflow.add_edge("longitudinal_sanity_check", "profile_update")
    workflow.add_edge("profile_update",           END)

    return workflow.compile()
