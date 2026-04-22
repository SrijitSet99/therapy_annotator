from typing import Dict, Any
from annotate.llm.ollama_client import call_ollama
from annotate.llm.json_parser import repair_json
from annotate.utils.prompt_loader import format_prompt, load_prompt
from annotate.utils.logger import log
from annotate.schemas.clinical_schema import AdviceOutput
from annotate.utils.text_file_logger import TextFileLogger        # ← ADD

logger = TextFileLogger()                                # ← ADD (module-level)
def advice_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract cessation advice."""
    log("advice_agent: extracting advice …")
    prompt = format_prompt(load_prompt("advice.txt"), conversation=state["chat"])
    logger.save("advice", "prompt", prompt)
    response = call_ollama(prompt)
    logger.save("advice", "response", response)
    parsed = repair_json(response)
    try:
        validated = AdviceOutput(**parsed)
    except Exception as exc:
        log(f"advice_agent: validation warning — {exc}. Using raw parsed output.")
        validated = AdviceOutput()
    log(f"advice_agent: found {len(validated.advice)} advice item(s).")
    return {"advice": validated.model_dump()}
