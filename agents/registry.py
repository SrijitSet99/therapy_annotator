"""
agents/registry.py — robust registry with multi-folder discovery,
strict execution, and safer parallel handling
"""
import importlib
import inspect
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Callable, List, Optional

from annotate.utils.logger import log, warning

AgentFn = Callable[[Dict[str, Any]], Dict[str, Any]]

# folders to auto-discover
_AGENT_FOLDERS = ["extractors", "confidence"]

_SKIP_AGENTS = set()


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: Dict[str, AgentFn] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 🔍 Discovery (FIXED: multi-folder support)
    # ------------------------------------------------------------------
    def discover(self, skip: set = _SKIP_AGENTS) -> None:
        base_dir = Path(__file__).resolve().parent

        total = 0
        for folder in _AGENT_FOLDERS:
            folder_path = base_dir / folder

            if not folder_path.exists():
                warning(f"registry: folder '{folder}' not found — skipping")
                continue

            for py_file in sorted(folder_path.glob("*.py")):
                if py_file.stem.startswith("_") or py_file.stem in skip:
                    continue

                module_path = f"annotate.agents.{folder}.{py_file.stem}"

                try:
                    module = importlib.import_module(module_path)
                except Exception as exc:
                    warning(f"registry: import failed {module_path}: {exc}")
                    continue

                # Register all callable functions in the module that are agents
                for attr_name in dir(module):
                    if attr_name.startswith("_") or not attr_name.endswith("_agent"):
                        continue
                    
                    fn = getattr(module, attr_name, None)
                    if fn is None or not callable(fn) or not hasattr(fn, '__name__'):
                        continue

                    # Skip if it's not a function (e.g., classes, modules)
                    if not inspect.isfunction(fn):
                        continue

                    if len(inspect.signature(fn).parameters) < 1:
                        continue

                    self._agents[attr_name] = fn
                    total += 1
                    log(f"registry: registered '{attr_name}' from {folder}")

        log(f"registry: total {total} agent(s) registered.")

    # ------------------------------------------------------------------
    # 🧩 Manual registration
    # ------------------------------------------------------------------
    def register(self, name: str, fn: AgentFn, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._agents[name] = fn
        if metadata:
            self._metadata[name] = metadata
        log(f"registry: manually registered '{name}'")

    def unregister(self, name: str) -> None:
        self._agents.pop(name, None)
        self._metadata.pop(name, None)

    @property
    def agent_names(self) -> List[str]:
        return list(self._agents.keys())

    # ------------------------------------------------------------------
    # 🎯 Single agent execution
    # ------------------------------------------------------------------
    def run_one(self, name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single agent by name."""
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not found. Available: {self.agent_names}")
        return self._agents[name](state)

    # ------------------------------------------------------------------
    # 🎯 Selection (unchanged)
    # ------------------------------------------------------------------
    def select_agents(self, state: Dict[str, Any], min_relevance: float = 0.0) -> List[str]:
        if not self._metadata:
            return self.agent_names

        selected = [
            n for n in self._agents
            if self._metadata.get(n, {}).get("relevance", 1.0) >= min_relevance
        ]

        log(f"registry: selected {len(selected)}/{len(self._agents)} agents")
        return selected

    # ------------------------------------------------------------------
    # ⚡ Safe execution helpers
    # ------------------------------------------------------------------
    def _safe_execute(self, name: str, fn: AgentFn, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            output = fn(state)

            # ✅ enforce dict output
            if not isinstance(output, dict):
                raise ValueError(f"{name} returned non-dict output")

            # ✅ prevent empty silent failure
            if not output:
                warning(f"registry: '{name}' returned empty output")

            return output

        except Exception as exc:
            # ❗ HARD FAIL instead of silent corruption
            raise RuntimeError(f"Agent '{name}' failed: {exc}")

    # ------------------------------------------------------------------
    # 🧪 Sequential (debug-friendly)
    # ------------------------------------------------------------------
    def run_all_sequential(self, state: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}

        for name, fn in self._agents.items():
            log(f"registry: running '{name}' (sequential) ...")

            output = self._safe_execute(name, fn, {**state, **merged})
            merged.update(output)

        return merged

    # ------------------------------------------------------------------
    # 🚀 Parallel (FIXED: no silent failures)
    # ------------------------------------------------------------------
    def run_all_parallel(
        self,
        state: Dict[str, Any],
        max_workers: int = 5,
        selected: Optional[List[str]] = None,
        fail_fast: bool = True,
    ) -> Dict[str, Any]:

        agents_to_run = {
            n: fn for n, fn in self._agents.items()
            if selected is None or n in selected
        }

        merged: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {
                executor.submit(self._safe_execute, name, fn, state): name
                for name, fn in agents_to_run.items()
            }

            for future in as_completed(future_to_name):
                name = future_to_name[future]

                try:
                    result = future.result()
                    merged.update(result)
                    log(f"registry: '{name}' completed.")

                except Exception as exc:
                    errors[name] = str(exc)
                    warning(f"registry: '{name}' failed: {exc}")

                    if fail_fast:
                        raise RuntimeError(f"Parallel execution stopped due to failure in '{name}'")

        if errors:
            warning(f"registry: {len(errors)} agent(s) failed: {list(errors.keys())}")

        return merged


# ----------------------------------------------------------------------
# Singleton
# ----------------------------------------------------------------------
_default_registry: Optional[AgentRegistry] = None


def get_registry(auto_discover: bool = True) -> AgentRegistry:
    global _default_registry

    if _default_registry is None:
        _default_registry = AgentRegistry()

        if auto_discover:
            _default_registry.discover()

    return _default_registry
