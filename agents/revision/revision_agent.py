"""
agents/revision/revision_agent.py
---------------------------------
Revision logic to improve low-confidence extractions.
Uses signal-specific prompts from prompts/ folder.
"""
from typing import Dict, Any
from annotate.llm.ollama_client import call_ollama
from annotate.llm.json_parser import repair_json
from annotate.utils.logger import log
from annotate.utils.text_file_logger import TextFileLogger
from annotate.utils.prompt_loader import format_prompt, load_prompt

logger = TextFileLogger()

# Signal-specific revision prompts mapping
_SIGNAL_PROMPT_MAP = {
    "advice": "advice_revision.txt",
    "triggers": "trigger_revision.txt",
    "attempt": None,  # Not yet available
    "concern": None,  # Not yet available
    "readiness": None,  # Not yet available
}

# Mapping of signal names to their output variable names in the prompts
_PROMPT_OUTPUT_VAR_MAP = {
    "advice": "original_output",
    "triggers": "original_output",
    "attempt": "current_output",
    "concern": "current_output",
    "readiness": "current_output",
}

# Mapping of signal names to their feedback variable names in the prompts
_PROMPT_FEEDBACK_VAR_MAP = {
    "advice": "reviewer_feedback",
    "triggers": "reviewer_feedback",
    "attempt": "critic_feedback",
    "concern": "critic_feedback",
    "readiness": "critic_feedback",
}

# Fallback prompt
_DEFAULT_REVISION_PROMPT = """You are a clinical annotation reviser specializing in tobacco cessation data extraction.

ORIGINAL CONVERSATION:
{conversation}

ORIGINAL EXTRACTION:
{extracted_signal}

REVIEWER FEEDBACK:
{feedback}

Produce a revised extraction that addresses all reviewer feedback:
- Remove any hallucinated items.
- Add any omitted items that are genuinely present in the conversation.
- Keep the same JSON field structure.
- Return ONLY valid JSON.

Respond ONLY with valid JSON matching the original structure.
"""


def run_revision(
    signal_name: str,
    current_output: Dict[str, Any],
    conversation: str,
    feedback: str,
) -> Dict[str, Any]:
    """Revises an extraction based on confidence feedback."""
    log(f"revision: revising '{signal_name}' based on feedback...")

    # Try to load signal-specific prompt
    prompt_filename = _SIGNAL_PROMPT_MAP.get(signal_name)
    prompt = None
    
    if prompt_filename:
        try:
            prompt = load_prompt(prompt_filename)
            # Use the correct variable names for this signal's prompt
            output_var = _PROMPT_OUTPUT_VAR_MAP.get(signal_name, "current_output")
            feedback_var = _PROMPT_FEEDBACK_VAR_MAP.get(signal_name, "critic_feedback")
            
            prompt = format_prompt(
                prompt,
                conversation=conversation[:4000],
                **{output_var: current_output, feedback_var: feedback}
            )
            logger.save("revision", f"prompt_{signal_name}", prompt)
        except Exception as e:
            log(f"revision: failed to load signal-specific prompt '{prompt_filename}': {e}")
            prompt = None
    
    # Fall back to default prompt if signal-specific doesn't exist or failed to load
    if prompt is None:
        prompt = _DEFAULT_REVISION_PROMPT.format(
            conversation=conversation[:4000],
            extracted_signal=current_output,
            feedback=feedback
        )
        logger.save("revision", f"prompt_{signal_name}", prompt)
    
    raw = call_ollama(prompt)
    logger.save("revision", f"response_{signal_name}", raw)
    parsed = repair_json(raw)
    
    return parsed
