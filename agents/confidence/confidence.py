"""
agents/confidence/confidence.py
-----------------------------
Evaluation logic to verify extraction accuracy against the source text.
"""
from typing import Dict, Any, List
from annotate.llm.ollama_client import call_ollama
from annotate.llm.json_parser import repair_json
from annotate.utils.logger import log
from annotate.schemas.clinical_schema import ConfidenceResult
from annotate.utils.text_file_logger import TextFileLogger
from annotate.utils.prompt_loader import format_prompt, load_prompt

logger = TextFileLogger()

# Signal-specific confidence check prompts mapping
_SIGNAL_PROMPT_MAP = {
    "attempt": "attempt_confidence_check.txt",
    "readiness": "readiness_confidence_check.txt",
    "concern": "concern_confidence_check.txt",
    "triggers": "trigger_confidence_check.txt",
    "advice": None,  # No specific prompt for advice yet
}

# Mapping of signal names to their output variable names in the prompts
_PROMPT_OUTPUT_VAR_MAP = {
    "attempt": "attempt_output",
    "readiness": "trigger_output",  # Note: readiness prompt incorrectly uses trigger_output
    "concern": "concern_output",
    "triggers": "trigger_output",
    "advice": "extracted_signal",
}

# Fallback prompt
_DEFAULT_CONFIDENCE_PROMPT = """You are a very strict clinical data validator. Compare the EXTRACTED DATA against the CONVERSATION.

CONVERSATION:
{conversation}

EXTRACTED DATA ({signal_name}):
{extracted_signal}

Your task:
1. Very Strictly Check for hallucinations (data not in the text).
2. Very Strictly Check for omissions (important data missed).
3. Assign a confidence score between 0.0 and 1.0.

Respond ONLY with valid JSON:
{{
  "confidence": 0.0,
  "issues": ["<issue 1>", "..."]
}}
"""

signal_map = {
    "attempt": "previous smoking quit attempt",
    "readiness": "readiness to quit smoking",
    "concern": "health or personal concern related to smoking",
    "triggers": "smoking triggers",
    "advice": "cessation advice or guidance"
}
def run_confidence_check(
    signal_name: str,
    extracted_signal: Dict[str, Any],
    conversation: str
) -> ConfidenceResult:
    """Evaluates the quality of an extraction to decide if debate is needed."""
    log(f"confidence_check: evaluating '{signal_name}'...")

    # Try to load signal-specific prompt
    prompt_filename = _SIGNAL_PROMPT_MAP.get(signal_name)
    prompt = None
    
    if prompt_filename:
        try:
            prompt = load_prompt(prompt_filename)
            # Use the correct output variable name for this signal's prompt
            output_var = _PROMPT_OUTPUT_VAR_MAP.get(signal_name, "extracted_signal")
            prompt = format_prompt(
                prompt,
                conversation=conversation[:4000],
                **{output_var: extracted_signal}
            )
            logger.save("confidence", f"prompt_{signal_name}", prompt)
        except Exception as e:
            log(f"confidence_check: failed to load signal-specific prompt '{prompt_filename}': {e}")
            prompt = None
    
    # Fall back to default prompt if signal-specific doesn't exist or failed to load
    if prompt is None:
        prompt = _DEFAULT_CONFIDENCE_PROMPT.format(
            conversation=conversation[:4000],
            signal_name=signal_map.get(signal_name),
            extracted_signal=extracted_signal
        )
        logger.save("confidence", f"prompt_{signal_name}", prompt)
    
    raw = call_ollama(prompt)
    logger.save("confidence", f"response_{signal_name}", raw)
    parsed = repair_json(raw)
    
    # Normalize confidence score from 0-10 to 0-1 if needed
    conf_score = float(parsed.get("confidence", 0.5))
    if conf_score > 1.0:
        conf_score = conf_score / 10.0
    conf_score = min(1.0, max(0.0, conf_score))  # Clamp to [0.0, 1.0]
    
    # Ensure issues are strings
    issues_raw = parsed.get("issues", [])
    issues: List[str] = []
    for issue in issues_raw:
        if isinstance(issue, dict):
            # Convert dict to string representation
            issues.append(str(issue))
        else:
            issues.append(str(issue))
    
    return ConfidenceResult(
        confidence=conf_score,
        issues=issues
    )