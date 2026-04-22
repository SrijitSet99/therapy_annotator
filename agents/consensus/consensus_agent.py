"""
agents/consensus/consensus_agent.py
-------------------------------------
Multi-voter consensus with arbitration.

Instead of a single consensus call, three independent LLM calls produce
three ConsensusVote objects. The arbitration step then:
  - computes agreement scores per field
  - returns the majority answer for high-agreement fields
  - calls an arbitration LLM for fields with low agreement

This is a major research upgrade: the system now quantifies its own
inter-agent disagreement and exposes it in the final DatasetRow.
"""
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from annotate.llm.ollama_client import call_ollama
from annotate.llm.json_parser import repair_json
from annotate.utils.logger import log, warning, log_data
from annotate.schemas.clinical_schema import ConsensusVote, ArbitrationResult
from annotate.config.settings import CONSENSUS_VOTERS, DISAGREEMENT_ARBITRATION_THRESHOLD


_CONSENSUS_PROMPT = """You are a clinical annotation specialist (voter {voter_id} of {total_voters}).

CONVERSATION:
{conversation}

EXTRACTED SIGNALS:
- Advice: {advice}
- Quit Attempts: {attempts}
- Readiness / Stage: {readiness}
- Concerns: {concerns}
- Triggers: {triggers}

Based on ALL the above signals, produce a unified clinical annotation.
Be independent — do not anchor to previous voters.

Respond ONLY with valid JSON:
{{
  "stage_of_change": "precontemplation|contemplation|preparation|action|maintenance|relapse",
  "main_concern": "<single most prominent barrier>",
  "primary_trigger": "<most significant smoking trigger>",
  "quit_attempt_history": "<concise summary of prior attempts>",
  "recommended_intervention": ["<strategy 1>", "<strategy 2>", "..."]
}}
"""

_ARBITRATION_PROMPT = """You are a clinical arbitrator. Multiple annotators disagreed on these fields.

CONVERSATION (excerpt):
{conversation_excerpt}

DISAGREEING VOTES:
{votes_summary}

Produce the single best answer for the disputed fields.
Respond ONLY with valid JSON matching the field names above.
"""


def _single_vote(voter_id: int, total: int, state: Dict[str, Any]) -> ConsensusVote:
    prompt = _CONSENSUS_PROMPT.format(
        voter_id=voter_id,
        total_voters=total,
        conversation=state["chat"],
        advice=state.get("advice", {}),
        attempts=state.get("attempt", {}),
        readiness=state.get("readiness", {}),
        concerns=state.get("concern", {}),
        triggers=state.get("triggers", {}),
    )
    raw = call_ollama(prompt)
    parsed = repair_json(raw)
    try:
        vote = ConsensusVote(voter_id=f"voter_{voter_id}", **parsed)
    except Exception as exc:
        warning(f"consensus voter {voter_id}: parse warning — {exc}")
        vote = ConsensusVote(
            voter_id=f"voter_{voter_id}",
            stage_of_change=parsed.get("stage_of_change", "unknown"),
            main_concern=parsed.get("main_concern", "unknown"),
            primary_trigger=parsed.get("primary_trigger", "unknown"),
            quit_attempt_history=parsed.get("quit_attempt_history", "unknown"),
            recommended_intervention=parsed.get("recommended_intervention", []),
        )
    return vote


def _majority(values: List[str]) -> str:
    if not values:
        return "unknown"
    return max(set(values), key=values.count)


def _agreement_score(values: List[str]) -> float:
    if not values:
        return 0.0
    top_count = max(values.count(v) for v in set(values))
    return top_count / len(values)


def _arbitrate(field_name: str, votes: List[ConsensusVote], conversation: str) -> str:
    votes_summary = "\n".join(
        f"  Voter {v.voter_id}: {getattr(v, field_name)}" for v in votes
    )
    prompt = _ARBITRATION_PROMPT.format(
        conversation_excerpt=conversation[:800],
        votes_summary=f"Field: {field_name}\n{votes_summary}",
    )
    raw = call_ollama(prompt)
    parsed = repair_json(raw)
    return str(parsed.get(field_name, votes[0].__dict__[field_name]))


def consensus_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run CONSENSUS_VOTERS independent votes in parallel, then arbitrate
    on any field where vote agreement is below the threshold.
    """
    n = CONSENSUS_VOTERS
    log(f"consensus_agent: collecting {n} independent votes ...")

    votes: List[ConsensusVote] = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = {ex.submit(_single_vote, i + 1, n, state): i for i in range(n)}
        for future in as_completed(futures):
            try:
                votes.append(future.result())
            except Exception as exc:
                warning(f"consensus_agent: a vote failed — {exc}")

    if not votes:
        warning("consensus_agent: all votes failed — returning empty consensus.")
        return {"consensus": {}}

    log(f"consensus_agent: {len(votes)} votes collected.")

    # ------------------------------------------------------------------
    # Compute agreement per field; arbitrate where agreement is low
    # ------------------------------------------------------------------
    fields = ["stage_of_change", "main_concern", "primary_trigger", "quit_attempt_history"]
    result: Dict[str, Any] = {}
    dissenting: List[str] = []
    agreement_scores: List[float] = []

    for field in fields:
        values = [str(getattr(v, field)) for v in votes]
        score = _agreement_score(values)
        agreement_scores.append(score)
        log(f"consensus_agent: field={field}, agreement={score:.2f}, values={values}")

        if score >= DISAGREEMENT_ARBITRATION_THRESHOLD:
            result[field] = _majority(values)
        else:
            log(f"consensus_agent: low agreement on '{field}' — arbitrating ...")
            dissenting.append(field)
            result[field] = _arbitrate(field, votes, state["chat"])

    # Merge recommended_intervention by union of all voters (preserving order)
    seen: set = set()
    merged_interventions: List[str] = []
    for v in votes:
        for item in v.recommended_intervention:
            if item not in seen:
                seen.add(item)
                merged_interventions.append(item)
    result["recommended_intervention"] = merged_interventions

    overall_agreement = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.0
    result["vote_agreement"] = round(overall_agreement, 3)
    result["dissenting_fields"] = dissenting

    log(f"consensus_agent: overall_agreement={overall_agreement:.3f}, dissenting={dissenting}")
    payload = {"consensus": result}
    log_data("consensus_agent.output", payload)
    return payload
