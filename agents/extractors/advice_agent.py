from annotate.llm.ollama_client import call_ollama  # noqa: F401  (patched in tests)
from annotate.schemas.clinical_schema import AdviceOutput
from annotate.agents.extractors._base import make_extractor

advice_agent = make_extractor(
    name="advice",
    prompt_file="advice.txt",
    schema=AdviceOutput,
    state_key="advice",
    summary=lambda v: f"found {len(v.advice)} advice item(s).",
    module_name=__name__,
)
