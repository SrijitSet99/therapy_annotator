"""Factory for the per-signal extractor agents.

All five extractors share the same shape: load a prompt, call the LLM, parse
JSON, validate against a Pydantic schema, return one state key. The only
per-signal differences are the prompt filename, schema class, state key, and
a small summary log line.
"""
import sys
from typing import Dict, Any, Callable, Type

from pydantic import BaseModel

from annotate.llm.json_parser import repair_json
from annotate.utils.prompt_loader import format_prompt, load_prompt
from annotate.utils.logger import log
from annotate.utils.text_file_logger import TextFileLogger

_logger = TextFileLogger()


def make_extractor(
    name: str,
    prompt_file: str,
    schema: Type[BaseModel],
    state_key: str,
    summary: Callable[[BaseModel], str],
    module_name: str,
):
    def agent(state: Dict[str, Any]) -> Dict[str, Any]:
        log(f"{name}_agent: extracting …")
        prompt = format_prompt(load_prompt(prompt_file), conversation=state["chat"])
        _logger.save(name, "prompt", prompt)
        # Resolve call_ollama from the agent module's namespace so unittest.mock
        # patches against `annotate.agents.extractors.<name>_agent.call_ollama` apply.
        call_ollama = sys.modules[module_name].call_ollama
        response = call_ollama(prompt)
        _logger.save(name, "response", response)
        try:
            validated = schema(**repair_json(response))
        except Exception as exc:
            log(f"{name}_agent: validation warning — {exc}. Using defaults.")
            validated = schema()
        log(f"{name}_agent: {summary(validated)}")
        return {state_key: validated.model_dump()}

    agent.__name__ = f"{name}_agent"
    return agent
