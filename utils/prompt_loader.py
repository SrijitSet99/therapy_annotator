from pathlib import Path
import json
import re

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_PROMPTS_JSON_PATH = _PROMPTS_DIR / "prompts.json"

# Load prompts once on module import
_PROMPTS_CACHE = None


def _load_prompts_cache():
    """Load all prompts from the JSON file."""
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is None:
        if not _PROMPTS_JSON_PATH.exists():
            raise FileNotFoundError(f"Prompts file not found: {_PROMPTS_JSON_PATH}")
        with open(_PROMPTS_JSON_PATH, 'r', encoding='utf-8') as f:
            _PROMPTS_CACHE = json.load(f)
    return _PROMPTS_CACHE


def format_prompt(prompt: str, **kwargs) -> str:
    """Format only known placeholders in a prompt template.

    This avoids interpreting literal JSON braces that appear in prompt examples.
    """
    if not kwargs:
        return prompt

    pattern = re.compile(r"\{(" + "|".join(re.escape(key) for key in kwargs.keys()) + r")\}")

    def replace(match):
        key = match.group(1)
        if key not in kwargs:
            raise KeyError(f"Missing prompt placeholder: {key}")
        return str(kwargs[key])

    return pattern.sub(replace, prompt)


def load_prompt(filename: str) -> str:
    """Load a prompt from the consolidated prompts.json.

    Supports both the old flat filenames (for example, "concern.txt") and
    explicit nested keys (for example, "extractors.concern").
    """
    prompts = _load_prompts_cache()

    prompt = _lookup_prompt(prompts, filename)
    if prompt is not None:
        return prompt

    key = Path(filename).stem
    prompt = _lookup_prompt(prompts, key)
    if prompt is not None:
        return prompt

    available = ", ".join(_flatten_prompt_keys(prompts))
    raise FileNotFoundError(
        f"Prompt not found: {filename}. Available prompts: {available}"
    )


def _lookup_prompt(prompts, key: str):
    if key in prompts and isinstance(prompts[key], str):
        return prompts[key]

    if "." in key:
        node = prompts
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node if isinstance(node, str) else None

    for value in prompts.values():
        if isinstance(value, dict) and key in value and isinstance(value[key], str):
            return value[key]

    return None


def _flatten_prompt_keys(prompts, prefix: str = ""):
    keys = []
    for key, value in prompts.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys.extend(_flatten_prompt_keys(value, full_key))
        else:
            keys.append(full_key)
    return sorted(keys)
