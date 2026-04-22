"""
agents/memory/agent_memory.py
------------------------------
Lightweight in-process memory for tracking per-agent performance over time.

Stores:
  - per-agent confidence history (rolling window)
  - per-agent run count and failure count
  - case-level memory: previous conversation outcomes for similarity lookup

In a production system this would persist to Redis/SQLite. For the research
prototype it lives in-memory and can optionally be serialised to JSON.
"""
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Any, Optional, List

from annotate.utils.logger import log

_WINDOW = 50  # rolling window size for accuracy tracking


class AgentMemory:
    """
    Tracks per-agent reliability and stores case-level outcomes.
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        # confidence_history[agent_name] = deque of confidence floats
        self._confidence: Dict[str, deque] = defaultdict(lambda: deque(maxlen=_WINDOW))
        # failure_count[agent_name] = int
        self._failures: Dict[str, int] = defaultdict(int)
        # run_count[agent_name] = int
        self._runs: Dict[str, int] = defaultdict(int)
        # case_memory: list of {"conversation_id": ..., "dataset_row": ...}
        self._cases: List[Dict[str, Any]] = []
        self._persist_path = Path(persist_path) if persist_path else None

        if self._persist_path and self._persist_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Per-agent tracking
    # ------------------------------------------------------------------

    def record_run(self, agent_name: str, confidence: float, failed: bool = False) -> None:
        self._runs[agent_name] += 1
        self._confidence[agent_name].append(confidence)
        if failed:
            self._failures[agent_name] += 1
        log(f"memory: recorded run for '{agent_name}' (conf={confidence:.2f}, failed={failed})")

    def get_accuracy(self, agent_name: str) -> Optional[float]:
        """
        Returns mean confidence over the rolling window, or None if no data.
        Used as a proxy for accuracy until a gold-label evaluation is wired in.
        """
        history = self._confidence.get(agent_name)
        if not history:
            return None
        return sum(history) / len(history)

    def get_failure_rate(self, agent_name: str) -> float:
        runs = self._runs[agent_name]
        if runs == 0:
            return 0.0
        return self._failures[agent_name] / runs

    def agent_stats(self) -> Dict[str, Dict[str, Any]]:
        stats = {}
        for name in self._runs:
            stats[name] = {
                "runs": self._runs[name],
                "failures": self._failures[name],
                "failure_rate": round(self.get_failure_rate(name), 3),
                "mean_confidence": round(self.get_accuracy(name) or 0.0, 3),
            }
        return stats

    # ------------------------------------------------------------------
    # Case memory
    # ------------------------------------------------------------------

    def store_case(self, conversation_id: str, dataset_row: Dict[str, Any]) -> None:
        self._cases.append({"conversation_id": conversation_id, "dataset_row": dataset_row})
        if self._persist_path:
            self._save()

    def retrieve_similar_cases(self, stage: str, n: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve up to n past cases with the same stage_of_change.
        Simple exact-match lookup — can be replaced with embedding similarity.
        """
        matched = [
            c for c in self._cases
            if c["dataset_row"].get("stage_of_change") == stage
        ]
        return matched[-n:]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save(self) -> None:
        data = {
            "confidence": {k: list(v) for k, v in self._confidence.items()},
            "failures": dict(self._failures),
            "runs": dict(self._runs),
            "cases": self._cases[-200:],  # keep last 200
        }
        self._persist_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        data = json.loads(self._persist_path.read_text(encoding="utf-8"))
        for k, v in data.get("confidence", {}).items():
            self._confidence[k] = deque(v, maxlen=_WINDOW)
        self._failures.update(data.get("failures", {}))
        self._runs.update(data.get("runs", {}))
        self._cases = data.get("cases", [])
        log(f"memory: loaded {len(self._cases)} cases from {self._persist_path}")
