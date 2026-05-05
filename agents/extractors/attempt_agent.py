from annotate.llm.ollama_client import call_ollama  # noqa: F401  (patched in tests)
from annotate.schemas.clinical_schema import AttemptOutput
from annotate.agents.extractors._base import make_extractor

attempt_agent = make_extractor(
    name="attempt",
    prompt_file="attempt.txt",
    schema=AttemptOutput,
    state_key="attempt",
    summary=lambda v: f"found {len(v.attempts)} attempt(s).",
    module_name=__name__,
)
