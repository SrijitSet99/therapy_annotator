from typing import Dict, Any

from annotate.utils.logger import log
from annotate.schemas.clinical_schema import DebateRound, DebateResult
from annotate.utils.text_file_logger import TextFileLogger
from annotate.agents.confidence.confidence import run_confidence_check
from annotate.agents.revision.revision_agent import run_revision as revise_signal

logger = TextFileLogger()


def _extract_issues(suggestions) -> str:
    if not suggestions:
        return "No specific suggestions provided."
    if isinstance(suggestions, list):
        return "\n".join(suggestions) if suggestions else "No specific suggestions provided."
    if isinstance(suggestions, dict):
        issues = suggestions.get("issues", [])
        return "\n".join(issues) if issues else "No specific suggestions provided."
    return str(suggestions)


signal_map = {
    "attempt": "previous smoking quit attempt",
    "readiness": "readiness to quit smoking",
    "concern": "health or personal concern related to smoking",
    "triggers": "smoking triggers",
    "advice": "cessation advice or guidance"
}


def run_debate(
    signal_name: str,
    initial_output: Dict[str, Any],
    conversation: str,
    confidence_threshold: float,
    registry=None,
    confidence_suggestions: Dict[str, Any] = None,
) -> DebateResult:

    log(f"debate_loop [{signal_name}]: starting (threshold={confidence_threshold:.2f}) ...")

    current_output = initial_output
    rounds = []
    round_num = 0

    feedback_str = _extract_issues(confidence_suggestions)

    while True:
        round_num += 1
        log(f"debate_loop [{signal_name}]: round {round_num}")

        # -------- Revision --------
        try:
            revised = revise_signal(
                signal_name=signal_name,
                current_output=current_output,
                conversation=conversation,
                feedback=feedback_str,
            )
        except Exception as e:
            log(f"debate_loop [{signal_name}]: revision agent failed: {e}")
            revised = current_output

        if isinstance(revised, dict) and signal_name in revised and isinstance(revised[signal_name], dict):
            revised = revised[signal_name]

        if not revised:
            log(f"debate_loop [{signal_name}]: empty revision — keeping previous.")
            revised = current_output

        # -------- Confidence --------
        try:
            revised_eval = run_confidence_check(signal_name, revised, conversation)
        except Exception as e:
            log(f"debate_loop [{signal_name}]: confidence agent failed: {e}")
            break  # cannot continue without confidence

        # Defensive handling
        confidence_score = getattr(revised_eval, "confidence", 0.0)
        suggestions = getattr(revised_eval, "suggestions", None)

        crossed_threshold = confidence_score >= confidence_threshold

        # Update feedback
        new_feedback = _extract_issues(suggestions)
        if new_feedback != "No specific suggestions provided.":
            feedback_str = new_feedback

        rounds.append(DebateRound(
            round_number=round_num,
            proposer_output=current_output,
            critic_feedback=feedback_str,
            revised_output=revised,
            agreement_reached=crossed_threshold,
        ))

        current_output = revised

        if crossed_threshold:
            log(
                f"debate_loop [{signal_name}]: confidence="
                f"{confidence_score:.2f} >= {confidence_threshold:.2f} "
                f"(round {round_num})"
            )
            return DebateResult(
                agent_name=signal_name,
                rounds=rounds,
                final_output=current_output,
                total_rounds=round_num,
                converged=True,
            )

    # -------- Fallback Exit --------
    log(f"debate_loop [{signal_name}]: stopped before convergence because confidence could not be evaluated.")

    return DebateResult(
        agent_name=signal_name,
        rounds=rounds,
        final_output=current_output,
        total_rounds=round_num,
        converged=False,
    )


def maybe_debate(
    signal_name: str,
    extractor_output: Dict[str, Any],
    conversation: str,
    confidence_threshold: float,
    confidence_score: float,
    registry=None,
    confidence_suggestions: Dict[str, Any] = None,
) -> Dict[str, Any]:

    if confidence_score >= confidence_threshold:
        log(
            f"debate [{signal_name}]: confidence={confidence_score:.2f} >= threshold — skipping."
        )
        extractor_output["_debate_rounds"] = 0
        extractor_output["_debate_converged"] = True
        return extractor_output

    log(
        f"debate [{signal_name}]: confidence={confidence_score:.2f} < threshold — entering loop."
    )

    result = run_debate(
        signal_name=signal_name,
        initial_output=extractor_output,
        conversation=conversation,
        confidence_threshold=confidence_threshold,
        registry=registry,
        confidence_suggestions=confidence_suggestions,
    )

    final = result.final_output
    final["_debate_rounds"] = result.total_rounds
    final["_debate_converged"] = result.converged

    return final
