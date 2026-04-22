"""
graph/parallel_workflow.py
---------------------------
Alternative entry point that bypasses LangGraph and calls the AgentManager
directly. Useful for testing or environments where LangGraph is unavailable.
All the same upgrades apply: debate, multi-consensus, sanity check.
"""
from typing import Dict, Any

from annotate.agents.manager.agent_manager import AgentManager
from annotate.agents.consensus.consensus_agent import consensus_agent
from annotate.agents.judge.sanity_check_agent import sanity_check_agent
from annotate.agents.reasoning.reasoning_agent import reasoning_agent
from annotate.utils.logger import log


def run_parallel_pipeline(chat: str, conversation_id: str = "", enable_debate: bool = False) -> Dict[str, Any]:
    manager = AgentManager()

    state: Dict[str, Any] = {
        "chat": chat,
        "advice": {}, "attempt": {}, "readiness": {}, "concern": {}, "triggers": {},
        "consensus": {}, "confidence_results": {},
        "sanity_check": {}, "dataset_row": {},
    }
    if conversation_id:
        state["conversation_id"] = conversation_id

    log("parallel_workflow: fan-out extraction + debate ...")
    state.update(manager.run_extractors(state, enable_debate=enable_debate))
    state["cross_signal_disagreement"] = manager.compute_cross_signal_disagreement(state)

    log("parallel_workflow: multi-voter consensus ...")
    state.update(consensus_agent(state))

    log("parallel_workflow: sanity check ...")
    state.update(sanity_check_agent(state))

    log("parallel_workflow: reasoning ...")
    state.update(reasoning_agent(state))

    return state["dataset_row"]
