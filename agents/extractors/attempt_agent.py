from typing import Dict, Any
from annotate.llm.ollama_client import call_ollama
from annotate.llm.json_parser import repair_json
from annotate.utils.prompt_loader import format_prompt, load_prompt
from annotate.utils.logger import log
from annotate.schemas.clinical_schema import AttemptOutput
from annotate.utils.text_file_logger import TextFileLogger        # ← ADD

logger = TextFileLogger()                                # ← ADD (module-level)


def attempt_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract quit attempt history."""
    log("attempt_agent: extracting quit attempt history …")
    prompt = format_prompt(load_prompt("attempt.txt"), conversation=state["chat"])
    logger.save("attempt", "prompt", prompt)             # ← ADD
    response = call_ollama(prompt)
    logger.save("attempt", "response", response)             # ← ADD
    parsed = repair_json(response)
    try:
        validated = AttemptOutput(**parsed)
    except Exception as exc:
        log(f"attempt_agent: validation warning — {exc}. Using defaults.")
        validated = AttemptOutput()
    log(f"attempt_agent: found {len(validated.attempts)} attempt(s).")
    return {"attempt": validated.model_dump()}
