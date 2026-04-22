from typing import Dict, Any
from annotate.llm.ollama_client import call_ollama
from annotate.llm.json_parser import repair_json
from annotate.utils.prompt_loader import format_prompt, load_prompt
from annotate.utils.logger import log
from annotate.schemas.clinical_schema import ReadinessOutput
from annotate.utils.text_file_logger import TextFileLogger        # ← ADD

logger = TextFileLogger()                                # ← ADD (module-level)
def readiness_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Classify TTM stage of change."""
    log("readiness_agent: classifying stage of change …")
    prompt = format_prompt(load_prompt("readiness.txt"), conversation=state["chat"])
    logger.save("readiness", "prompt", prompt)
    response = call_ollama(prompt)
    logger.save("readiness", "response", response)
    parsed = repair_json(response)
    try:
        validated = ReadinessOutput(**parsed)
    except Exception as exc:
        log(f"readiness_agent: validation warning — {exc}. Defaulting to 'unknown'.")
        validated = ReadinessOutput(readiness_stage="unknown")
    log(f"readiness_agent: stage={validated.readiness_stage}.")
    return {"readiness": validated.model_dump()}
