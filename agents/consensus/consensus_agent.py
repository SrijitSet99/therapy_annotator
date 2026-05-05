"""
agents/consensus/consensus_agent.py
-------------------------------------

Multi-voter consensus with arbitration.

Instead of relying on a single LLM output, this module:
1. Collects multiple independent "votes" from different LLM calls.
2. Measures agreement between those votes.
3. Uses majority voting when agreement is high.
4. Falls back to an arbitration LLM when agreement is low.

This improves robustness and provides transparency about disagreement.
"""

from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# LLM + utility imports
from annotate.llm.ollama_client import call_ollama
from annotate.llm.json_parser import repair_json
from annotate.utils.prompt_loader import format_prompt, load_prompt
from annotate.utils.logger import log, warning, log_data
from annotate.schemas.clinical_schema import ConsensusVote
from annotate.config.settings import CONSENSUS_VOTERS, DISAGREEMENT_ARBITRATION_THRESHOLD


# ---------------------------------------------------------------------
# Step 1: Generate a single vote from one "voter" (LLM call)
# ---------------------------------------------------------------------
def _single_vote(voter_id: int, total: int, state: Dict[str, Any]) -> ConsensusVote:
    """
    Each voter independently analyzes the conversation and produces a structured output.
    """

    # Prepare prompt with all required context
    prompt = format_prompt(
        load_prompt("consensus.consensus"),
        voter_id=voter_id,
        total_voters=total,
        conversation=state["chat"],
        advice=state.get("advice", {}),
        attempts=state.get("attempt", {}),
        readiness=state.get("readiness", {}),
        concerns=state.get("concern", {}),
        triggers=state.get("triggers", {}),
    )

    # Call LLM
    raw = call_ollama(prompt)

    # Attempt to repair malformed JSON from LLM output
    parsed = repair_json(raw)

    # Convert parsed JSON into a structured schema object
    try:
        vote = ConsensusVote(voter_id=f"voter_{voter_id}", **parsed)

    # Fallback in case parsing or schema validation fails
    except Exception as exc:
        warning(f"consensus voter {voter_id}: parse warning — {exc}")

        # Use safe defaults for missing fields
        vote = ConsensusVote(
            voter_id=f"voter_{voter_id}",
            stage_of_change=parsed.get("stage_of_change", "unknown"),
            all_concerns=parsed.get("all_concerns", []) or [],
            all_triggers=parsed.get("all_triggers", []) or [],
            quit_attempt_history=parsed.get("quit_attempt_history", "unknown"),
            recommended_intervention=parsed.get("recommended_intervention", []) or [],
        )

    return vote


# ---------------------------------------------------------------------
# Utility: Majority voting
# ---------------------------------------------------------------------
def _majority(values: List[str]) -> str:
    """
    Returns the most common value in the list.
    If empty, returns 'unknown'.
    """
    if not values:
        return "unknown"
    return max(set(values), key=values.count)


# ---------------------------------------------------------------------
# Utility: Agreement score
# ---------------------------------------------------------------------
def _agreement_score(values: List[str]) -> float:
    """
    Measures how much voters agree on a field.

    Example:
    ["A", "A", "B"] → agreement = 2/3 = 0.67
    """
    if not values:
        return 0.0

    top_count = max(values.count(v) for v in set(values))
    return top_count / len(values)


# ---------------------------------------------------------------------
# Step 2: Arbitration for low-agreement fields
# ---------------------------------------------------------------------
def _arbitrate(field_name: str, votes: List[ConsensusVote], conversation: str) -> str:
    """
    When voters disagree significantly, call another LLM to decide.

    This acts as a "judge" model.
    """

    # Create a summary of all voter opinions for this field
    votes_summary = "\n".join(
        f"  Voter {v.voter_id}: {getattr(v, field_name)}" for v in votes
    )

    # Build arbitration prompt
    prompt = format_prompt(
        load_prompt("consensus.arbitration"),
        conversation_excerpt=conversation[:800],  # limit context size
        votes_summary=f"Field: {field_name}\n{votes_summary}",
    )

    # Call arbitration LLM
    raw = call_ollama(prompt)
    parsed = repair_json(raw)

    # Return arbitrated value or fallback to first vote
    return str(parsed.get(field_name, votes[0].__dict__[field_name]))


# ---------------------------------------------------------------------
# Main: Consensus Agent
# ---------------------------------------------------------------------
def consensus_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core pipeline:

    1. Run multiple voters in parallel
    2. Measure agreement for each field
    3. Use:
        - majority vote (high agreement)
        - arbitration LLM (low agreement)
    4. Merge interventions
    5. Return structured consensus output
    """

    n = CONSENSUS_VOTERS
    log(f"consensus_agent: collecting {n} independent votes ...")

    votes: List[ConsensusVote] = []

    # Run voters in parallel for speed
    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = {ex.submit(_single_vote, i + 1, n, state): i for i in range(n)}

        for future in as_completed(futures):
            try:
                votes.append(future.result())
            except Exception as exc:
                warning(f"consensus_agent: a vote failed — {exc}")

    # If all votes fail, return empty result
    if not votes:
        warning("consensus_agent: all votes failed — returning empty consensus.")
        return {"consensus": {}}

    log(f"consensus_agent: {len(votes)} votes collected.")

    # ------------------------------------------------------------------
    # Step 3: Agreement + Arbitration (singleton string fields only)
    # ------------------------------------------------------------------
    fields = [
        "stage_of_change",
        "quit_attempt_history",
    ]

    result: Dict[str, Any] = {}
    dissenting: List[str] = []
    agreement_scores: List[float] = []

    for field in fields:
        # Collect all voter values for this field
        values = [str(getattr(v, field)) for v in votes]

        # Compute agreement score
        score = _agreement_score(values)
        agreement_scores.append(score)

        log(f"consensus_agent: field={field}, agreement={score:.2f}, values={values}")

        # High agreement → use majority
        if score >= DISAGREEMENT_ARBITRATION_THRESHOLD:
            result[field] = _majority(values)

        # Low agreement → call arbitration model
        else:
            log(f"consensus_agent: low agreement on '{field}' — arbitrating ...")
            dissenting.append(field)
            result[field] = _arbitrate(field, votes, state["chat"])

    # ------------------------------------------------------------------
    # Step 4: Union list-typed fields across voters (no arbitration)
    # ------------------------------------------------------------------
    # all_concerns, all_triggers, recommended_intervention are sourced from
    # extractor signals per the consensus prompt rules — voters should agree
    # on content, so we union (preserving order, dedupe case-insensitively).
    for list_field in ("all_concerns", "all_triggers", "recommended_intervention"):
        seen: set = set()
        merged: List[str] = []
        for v in votes:
            for item in getattr(v, list_field, []):
                key = item.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    merged.append(item.strip())
        result[list_field] = merged

    # ------------------------------------------------------------------
    # Step 5: Compute overall agreement
    # ------------------------------------------------------------------
    overall_agreement = (
        sum(agreement_scores) / len(agreement_scores)
        if agreement_scores else 0.0
    )

    result["vote_agreement"] = round(overall_agreement, 3)
    result["dissenting_fields"] = dissenting

    log(f"consensus_agent: overall_agreement={overall_agreement:.3f}, dissenting={dissenting}")

    # Final payload
    payload = {"consensus": result}

    log_data("consensus_agent.output", payload)

    return payload