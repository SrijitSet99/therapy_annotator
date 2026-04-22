"""
agents/judge/sanity_check_agent.py
------------------------------------
Post-hoc validation agent that catches logical inconsistencies in the
assembled annotation BEFORE the final DatasetRow is emitted.

Examples of inconsistencies caught:
  - stage=precontemplation but intervention="quit immediately"
  - stage=maintenance but concern="not ready to quit"
  - critical fields resolved to "unknown"
  - low voter agreement but high reported confidence
"""
from typing import Dict, Any, List
from annotate.utils.logger import log
from annotate.schemas.clinical_schema import SanityCheckResult


def _check_stage_intervention_conflict(state: Dict[str, Any]) -> List[str]:
    issues = []
    stage = state.get("consensus", {}).get("stage_of_change", "").lower()
    interventions = [i.lower() for i in state.get("consensus", {}).get("recommended_intervention", [])]
    aggressive_keywords = {"quit immediately", "set a quit date today", "start medication now"}
    passive_stages = {"precontemplation", "contemplation"}
    if stage in passive_stages:
        for kw in aggressive_keywords:
            if any(kw in iv for iv in interventions):
                issues.append(f"Stage '{stage}' is pre-action but intervention includes aggressive directive: '{kw}'")
    return issues


def _check_maintenance_stage_concerns(state: Dict[str, Any]) -> List[str]:
    issues = []
    stage = state.get("consensus", {}).get("stage_of_change", "").lower()
    concern = state.get("consensus", {}).get("main_concern", "").lower()
    not_ready_phrases = ["not ready", "don't want to quit", "not considering"]
    if stage == "maintenance":
        for phrase in not_ready_phrases:
            if phrase in concern:
                issues.append(f"Stage is 'maintenance' but main_concern implies unwillingness: '{concern}'")
    return issues


def _check_unknown_fields(state: Dict[str, Any]) -> List[str]:
    issues = []
    consensus = state.get("consensus", {})
    for field in ["stage_of_change", "main_concern", "primary_trigger"]:
        if consensus.get(field, "unknown").lower() in ("unknown", "", "none"):
            issues.append(f"Critical field '{field}' resolved to 'unknown' — extraction may have failed.")
    return issues


def _check_low_agreement_high_confidence(state: Dict[str, Any]) -> List[str]:
    issues = []
    agreement = state.get("consensus", {}).get("vote_agreement", 1.0)
    confidence = (state.get("dataset_row", {}).get("confidence_score") or 0.5)
    if agreement < 0.4 and confidence > 0.8:
        issues.append(
            f"Voter agreement is low ({agreement:.2f}) but confidence_score is high ({confidence:.2f}). "
            "Confidence may be overestimated."
        )
    return issues


_CHECKS = [
    _check_stage_intervention_conflict,
    _check_maintenance_stage_concerns,
    _check_unknown_fields,
    _check_low_agreement_high_confidence,
]


def sanity_check_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    log("sanity_check_agent: running consistency checks ...")
    all_issues: List[str] = []
    for check_fn in _CHECKS:
        try:
            all_issues.extend(check_fn(state))
        except Exception as exc:
            log(f"sanity_check_agent: check '{check_fn.__name__}' raised {exc}")

    passed = len(all_issues) == 0
    severity = "none" if passed else ("high" if len(all_issues) >= 3 else "medium")

    result = SanityCheckResult(passed=passed, inconsistencies=all_issues, severity=severity)
    if not passed:
        log(f"sanity_check_agent: FAILED — {len(all_issues)} issue(s) (severity={severity})")
        for issue in all_issues:
            log(f"  WARNING: {issue}")
    else:
        log("sanity_check_agent: all checks passed.")
    return {"sanity_check": result.model_dump()}
