from annotate.llm.ollama_client import call_ollama  # noqa: F401  (patched in tests)
from annotate.schemas.clinical_schema import ConcernOutput
from annotate.agents.extractors._base import make_extractor

concern_agent = make_extractor(
    name="concern",
    prompt_file="concern.txt",
    schema=ConcernOutput,
    state_key="concern",
    summary=lambda v: f"found {len(v.concerns)} concern(s).",
    module_name=__name__,
)
