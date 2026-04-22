from typing import Dict, Any
from annotate.llm.ollama_client import call_ollama
from annotate.llm.json_parser import repair_json
from annotate.utils.prompt_loader import format_prompt, load_prompt
from annotate.utils.logger import log
from annotate.schemas.clinical_schema import ConcernOutput
from annotate.utils.text_file_logger import TextFileLogger        # ← ADD

logger = TextFileLogger()                                # ← ADD (module-level)

def concern_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract patient concerns."""
    log("concern_agent: extracting patient concerns …")
    prompt = format_prompt(load_prompt("concern.txt"), conversation=state["chat"])
    logger.save("concern", "prompt", prompt)
    response = call_ollama(prompt)
    logger.save("concern", "response", response)
    parsed = repair_json(response)
    try:
        validated = ConcernOutput(**parsed)
    except Exception as exc:
        log(f"concern_agent: validation warning — {exc}. Using defaults.")
        validated = ConcernOutput()
    log(f"concern_agent: found {len(validated.concerns)} concern(s).")
    return {"concern": validated.model_dump()}
