import json
import os
from datetime import datetime
from typing import Any

from annotate.config.settings import LOG_DIR


class PromptLogger:
    """Logs all LLM prompts to a single timestamped text file per process."""

    def __init__(self, output_dir: str | None = None):
        output_dir = output_dir or LOG_DIR
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(output_dir, f"prompts_{timestamp}.txt")
        self._write(f"=== Prompt Log — {timestamp} ===\n\n")

    def save(self, agent: str, prompt: str) -> None:
        """
        Append a prompt to the log file.

        Args:
            agent:  name of the agent sending the prompt (e.g. 'attempt_agent')
            prompt: the full prompt string sent to the LLM
        """
        block = (
            f"--- [{agent.upper()}] "
            f"@ {datetime.now().strftime('%H:%M:%S.%f')[:-3]} ---\n"
            f"{prompt}\n"
            f"{'=' * 60}\n\n"
        )
        self._write(block)

    def _write(self, text: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text)

class TextFileLogger:
    """Appends all LLM responses to a single timestamped text file per run."""

    def __init__(self, output_dir: str | None = None):
        output_dir = output_dir or LOG_DIR
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(output_dir, f"llm_responses_{timestamp}.txt")
        self._write(f"=== LLM Response Log — {timestamp} ===\n\n")

    def save(self, stage: str, signal: str, data: Any) -> None:
        """
        Append one LLM response to the log file.

        Args:
            stage:  'extractor' | 'confidence' | 'debate'
            signal: the state_key for this signal (e.g. 'intent', 'sentiment')
            data:   the raw response object (dict, Pydantic model, str, etc.)
        """
        if hasattr(data, "model_dump"):
            serialised = data.model_dump()
        elif hasattr(data, "__dict__"):
            serialised = data.__dict__
        else:
            serialised = data

        block = (
            f"--- [{stage.upper()}] {signal} "
            f"@ {datetime.now().strftime('%H:%M:%S.%f')[:-3]} ---\n"
            f"{json.dumps(serialised, indent=2, default=str)}\n\n"
        )
        self._write(block)

    def _write(self, text: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text)
