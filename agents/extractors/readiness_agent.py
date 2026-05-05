from annotate.llm.ollama_client import call_ollama  # noqa: F401  (patched in tests)
from annotate.schemas.clinical_schema import ReadinessOutput
from annotate.agents.extractors._base import make_extractor

readiness_agent = make_extractor(
    name="readiness",
    prompt_file="readiness.txt",
    schema=ReadinessOutput,
    state_key="readiness",
    summary=lambda v: f"stage={v.readiness_stage}.",
    module_name=__name__,
)
