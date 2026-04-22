from typing import Dict, Any
from annotate.llm.ollama_client import call_ollama
from annotate.llm.json_parser import repair_json
from annotate.utils.prompt_loader import format_prompt, load_prompt
from annotate.utils.logger import log
from annotate.schemas.clinical_schema import TriggerOutput
from annotate.utils.text_file_logger import TextFileLogger        # ← ADD

logger = TextFileLogger()                                # ← ADD (module-level)
def trigger_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Identify smoking triggers"""
    log("trigger_agent: identifying smoking triggers …")
    prompt = format_prompt(load_prompt("trigger.txt"), conversation=state["chat"])
    logger.save("trigger", "prompt", prompt)
    response = call_ollama(prompt)
    logger.save("trigger", "response", response)
    parsed = repair_json(response)
    try:
        validated = TriggerOutput(**parsed)
    except Exception as exc:
        log(f"trigger_agent: validation warning — {exc}. Using defaults.")
        validated = TriggerOutput()
    log(f"trigger_agent: found {len(validated.triggers)} trigger(s).")
    return {"triggers": validated.model_dump()}
