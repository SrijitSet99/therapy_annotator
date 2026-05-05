from annotate.llm.ollama_client import call_ollama  # noqa: F401  (patched in tests)
from annotate.schemas.clinical_schema import TriggerOutput
from annotate.agents.extractors._base import make_extractor

trigger_agent = make_extractor(
    name="trigger",
    prompt_file="trigger.txt",
    schema=TriggerOutput,
    state_key="triggers",
    summary=lambda v: f"found {len(v.triggers)} trigger(s).",
    module_name=__name__,
)
