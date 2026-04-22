import time
import requests
from typing import Dict, Any

from annotate.config.model_config import MODEL_NAME, TEMPERATURE, OLLAMA_URL, MAX_RETRIES, TIMEOUT_SECONDS
from annotate.utils.logger import log, log_ollama_exchange


def call_ollama(prompt: str, model: str = MODEL_NAME, temperature: float = TEMPERATURE) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SECONDS)
            if response.status_code != 200:
                raise RuntimeError(f"Ollama HTTP {response.status_code}: {response.text[:5000]}")
            response_text = response.json()["response"]
            log_ollama_exchange(
                model=model,
                temperature=temperature,
                prompt=prompt,
                response_text=response_text,
                attempt=attempt,
                status="success",
            )
            return response_text

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = exc
            wait = 2 ** attempt
            log(f"[ollama_client] Attempt {attempt}/{MAX_RETRIES} failed. Retrying in {wait}s…")
            log_ollama_exchange(
                model=model,
                temperature=temperature,
                prompt=prompt,
                response_text=str(exc),
                attempt=attempt,
                status=exc.__class__.__name__,
            )
            time.sleep(wait)

        except Exception as exc:
            log_ollama_exchange(
                model=model,
                temperature=temperature,
                prompt=prompt,
                response_text=str(exc),
                attempt=attempt,
                status="exception",
            )
            raise RuntimeError(f"Ollama call failed: {exc}") from exc

    raise RuntimeError(f"Ollama unreachable after {MAX_RETRIES} attempts. Last error: {last_error}")
