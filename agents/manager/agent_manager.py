"""
agents/manager/agent_manager.py
---------------------------------
Central coordinator for parallel extractor fan-out and confidence-aware routing.

Flow:
  1. run_extractors()       — parallel fan-out, returns raw extractor outputs
  2. run_confidence_check() — evaluate every signal; called by the graph node
  3. run_revision()         — revise only signals below threshold; called by graph node
  Steps 2-3 repeat (workflow loop) until all signals pass the threshold
  or max revision rounds is reached. The threshold and max-rounds are pulled
  from state (set by run_pipeline based on enable_debate) with a fallback to
  config defaults.
"""
from statistics import pstdev
from typing import Dict, Any, List, Optional

from annotate.agents.registry import get_registry
from annotate.agents.memory.agent_memory import AgentMemory
from annotate.agents.confidence.confidence import run_confidence_check
from annotate.agents.revision.revision_agent import run_revision as revise_signal
from annotate.utils.logger import log, warning, log_data
from annotate.utils.text_file_logger import TextFileLogger
from annotate.config.settings import CONFIDENCE_DEBATE_THRESHOLD


# Map from registry key (function name) → signal key (state key)
_SIGNAL_MAP = {
    "attempt_agent":   "attempt",
    "readiness_agent": "readiness",
    "concern_agent":   "concern",
    "trigger_agent":   "triggers",
    "advice_agent":    "advice",
}

class AgentManager:
    """
    Coordinates extractor agents with confidence-aware routing and optional debate escalation.
    """

    def __init__(self, memory: Optional[AgentMemory] = None) -> None:
        self.registry = get_registry()
        self.memory = memory or AgentMemory()

    # ------------------------------------------------------------------
    # Step 1 — Parallel extraction (no confidence logic here)
    # ------------------------------------------------------------------

    def run_extractors(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fan out all registered extractors in parallel.
        Returns raw extractor outputs merged into a single dict.
        Confidence checking and revision are done in separate graph nodes
        (the workflow's confidence ↔ revision loop).
        """
        log("agent_manager: running parallel extractors ...")
        logger = TextFileLogger()

        # Only run main extraction agents, not revision agents
        main_agents = [name for name in self.registry.agent_names if not name.endswith("_revision_agent")]
        raw_outputs = self.registry.run_all_parallel(state, max_workers=5, selected=main_agents)

        final_outputs: Dict[str, Any] = {}
        for state_key in _SIGNAL_MAP.values():
            output = raw_outputs.get(state_key)
            if not output:
                continue
            log_data(f"agent_manager.extractor_output.{state_key}", output)
            logger.save("extractor", state_key, output)
            final_outputs[state_key] = output

        # confidence_results starts empty; will be filled by run_confidence_check()
        final_outputs["confidence_results"] = {}
        return final_outputs

    # ------------------------------------------------------------------
    # Step 2 — Confidence check (called by graph confidence_check node)
    # ------------------------------------------------------------------

    def run_confidence_check(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the confidence of every extracted signal.
        Returns {'confidence_results': {signal: {'confidence': float, 'issues': list}}}.
        Called directly by the graph confidence_check node.
        """
        log("agent_manager: running confidence checks ...")
        logger = TextFileLogger()

        confidence_results: Dict[str, Any] = {}
        chat_text = state.get("chat", "")

        for state_key in _SIGNAL_MAP.values():
            output = state.get(state_key)
            if not output:
                # Signal missing — treat as zero confidence so revision will be triggered
                confidence_results[state_key] = {"confidence": 0.0, "issues": ["signal not extracted"]}
                continue

            result = self._run_signal_confidence(state, state_key, output)
            confidence_results[state_key] = result
            logger.save("confidence_check", state_key, result)
            log(f"agent_manager: confidence[{state_key}]={result['confidence']:.3f} "
                f"issues={result['issues']}")

        log_data("agent_manager.confidence_results", confidence_results)
        return {"confidence_results": confidence_results}

    # ------------------------------------------------------------------
    # Step 3 — Revision (called by graph revision node)
    # ------------------------------------------------------------------

    def run_revision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Revise every signal whose confidence is below the threshold.
        Returns partial state update with revised signal values.
        Called directly by the graph revision node.
        """
        log("agent_manager: running revisions for low-confidence signals ...")
        logger = TextFileLogger()

        confidence_results = state.get("confidence_results", {})
        threshold = state.get("confidence_threshold", CONFIDENCE_DEBATE_THRESHOLD)
        revised_outputs: Dict[str, Any] = {}

        for state_key in _SIGNAL_MAP.values():
            conf_entry = confidence_results.get(state_key, {})
            confidence = conf_entry.get("confidence", 0.0)
            issues = conf_entry.get("issues", [])

            if confidence >= threshold:
                log(f"agent_manager: '{state_key}' confidence={confidence:.3f} >= threshold "
                    f"— skipping revision.")
                continue

            log(f"agent_manager: '{state_key}' confidence={confidence:.3f} < threshold "
                f"— running revision ...")

            current_output = state.get(state_key, {})

            try:
                revised = revise_signal(
                    signal_name=state_key,
                    current_output=current_output,
                    conversation=state.get("chat", ""),
                    feedback="\n".join(str(issue) for issue in issues) if issues else "No issues provided.",
                )
                if isinstance(revised, dict) and state_key in revised and isinstance(revised[state_key], dict):
                    revised = revised[state_key]

                if isinstance(revised, dict) and revised:
                    revised_outputs[state_key] = revised
                    logger.save("revision", state_key, revised)
                    log_data(f"agent_manager.revised_output.{state_key}", revised)
                else:
                    warning(f"agent_manager: revision for '{state_key}' returned unexpected keys: "
                            f"{list(revised.keys()) if isinstance(revised, dict) else type(revised)} — keeping original.")
                    revised_outputs[state_key] = current_output
            except Exception as exc:
                warning(f"agent_manager: revision for '{state_key}' failed: {exc}. Keeping original output.")
                revised_outputs[state_key] = current_output

        return revised_outputs

    # ------------------------------------------------------------------
    # Cross-signal disagreement metric
    # ------------------------------------------------------------------

    def compute_cross_signal_disagreement(self, state: Dict[str, Any]) -> float:
        """Compute standard deviation of per-signal confidence scores."""
        confidence_results = state.get("confidence_results", {})
        confidences: List[float] = []

        for state_key in _SIGNAL_MAP.values():
            confidence = confidence_results.get(state_key, {}).get("confidence")
            if confidence is not None:
                confidences.append(float(confidence))

        if len(confidences) < 2:
            return 0.0

        return round(pstdev(confidences), 4)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_signal_confidence(
        self,
        state: Dict[str, Any],
        state_key: str,
        output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run the confidence check for a single signal.
        Returns {'confidence': float, 'issues': list}.
        Falls back to {'confidence': 0.5, 'issues': []} on error.
        """
        try:
            chat_text = state.get("chat", "")
            result = run_confidence_check(
                signal_name=state_key,
                extracted_signal=output,
                conversation=chat_text
            )
            return {"confidence": result.confidence, "issues": result.issues}
        except Exception as exc:
            warning(f"agent_manager: confidence check for '{state_key}' failed: {exc}. "
                    f"Defaulting to 0.5.")
            return {"confidence": 0.5, "issues": [str(exc)]}
