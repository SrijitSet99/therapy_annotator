import json
import re
from typing import Dict, Any


def repair_json(text: str) -> Dict[str, Any]:
    """Extract and parse a JSON object from a raw LLM response."""
    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    for start_match in reversed(list(re.finditer(r"\{", text))):
        start = start_match.start()
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start: i + 1])
                    except Exception:
                        break
    return {}
