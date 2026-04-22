import logging
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

_logger = logging.getLogger("clinical_pipeline")
_file_handler: logging.Handler | None = None
_trace_path: Path | None = None
_trace_payload: dict[str, Any] = {}
_ollama_text_path: Path | None = None
_ollama_mirror_path: Path | None = None
_ollama_call_count: int = 0


def log(msg: str) -> None:
    _logger.info(msg)


def warning(msg: str) -> None:
    _logger.warning(msg)


def error(msg: str) -> None:
    _logger.error(msg)


def configure_run_logging(log_path: str | Path) -> Path:
    """
    Attach a per-run file logger so interactive sessions keep a durable trace.
    Replaces any previously attached file handler.
    """
    global _file_handler

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if _file_handler is not None:
        _logger.removeHandler(_file_handler)
        _file_handler.close()

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _logger.addHandler(handler)
    _file_handler = handler
    _logger.info(f"run_log: writing detailed trace to {path}")
    return path


def configure_run_trace(trace_path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    """
    Create a structured JSON trace file for the current run.
    Subsequent log_data(...) calls are mirrored into this document.
    """
    global _trace_path, _trace_payload

    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _trace_path = path
    _trace_payload = {
        "meta": _make_jsonable(metadata or {}),
        "events": {},
    }
    _write_trace_file()
    _logger.info(f"run_trace: writing structured trace to {path}")
    return path


def log_data(label: str, data: Any) -> None:
    """
    Log structured payloads in a readable single block for debugging.
    """
    try:
        formatted = json.dumps(data, indent=2, ensure_ascii=False, default=str, sort_keys=True)
    except Exception:
        formatted = repr(data)
    _logger.info(f"{label}:\n{formatted}")
    if _trace_path is not None:
        _set_trace_value(label, _make_jsonable(data))
        _write_trace_file()


def configure_ollama_text_log(text_path: str | Path) -> Path:
    """
    Configure a single text transcript file that records every Ollama prompt/response
    exchange for the current run.
    """
    global _ollama_text_path, _ollama_mirror_path, _ollama_call_count

    path = Path(text_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ollama_text_path = path
    _ollama_mirror_path = path.parent / "ollama_full_output.txt"
    _ollama_call_count = 0
    path.write_text("", encoding="utf-8")
    _ollama_mirror_path.write_text("", encoding="utf-8")
    _logger.info(f"run_ollama_text: writing full Ollama exchanges to {path}")
    _logger.info(f"run_ollama_text: mirroring latest run to {_ollama_mirror_path}")
    return path


def log_ollama_exchange(
    *,
    model: str,
    temperature: float,
    prompt: str,
    response_text: str,
    attempt: int,
    status: str = "success",
) -> Path:
    """
    Append a full Ollama request/response exchange to the current transcript file.
    Creates a default transcript file lazily if one has not been configured yet.
    """
    global _ollama_call_count

    path = _ensure_ollama_text_log()
    _ollama_call_count += 1
    timestamp = datetime.now(timezone.utc).isoformat()
    block = (
        f"\n{'=' * 80}\n"
        f"OLLAMA CALL {_ollama_call_count:04d}\n"
        f"{'=' * 80}\n"
        f"timestamp: {timestamp}\n"
        f"status: {status}\n"
        f"attempt: {attempt}\n"
        f"model: {model}\n"
        f"temperature: {temperature}\n"
        f"\nPROMPT:\n"
        f"{prompt}\n"
        f"\nRESPONSE:\n"
        f"{response_text}\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(block)
    if _ollama_mirror_path is not None:
        with _ollama_mirror_path.open("a", encoding="utf-8") as handle:
            handle.write(block)
    return path


def get_ollama_text_log_path() -> Path | None:
    return _ollama_text_path


def _make_jsonable(data: Any) -> Any:
    try:
        return json.loads(json.dumps(data, ensure_ascii=False, default=str))
    except Exception:
        return repr(data)


def _set_trace_value(label: str, data: Any) -> None:
    node = _trace_payload.setdefault("events", {})
    parts = [part for part in label.split(".") if part]
    if not parts:
        return
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = data


def _write_trace_file() -> None:
    if _trace_path is None:
        return
    _trace_path.write_text(
        json.dumps(_trace_payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _ensure_ollama_text_log() -> Path:
    global _ollama_text_path, _ollama_mirror_path, _ollama_call_count

    if _ollama_text_path is not None:
        return _ollama_text_path

    from annotate.config.settings import LOG_DIR

    run_dir = Path(LOG_DIR) / "runs"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _ollama_text_path = run_dir / f"ollama_calls_{stamp}.txt"
    _ollama_mirror_path = run_dir / "ollama_full_output.txt"
    _ollama_text_path.parent.mkdir(parents=True, exist_ok=True)
    _ollama_text_path.write_text("", encoding="utf-8")
    _ollama_mirror_path.write_text("", encoding="utf-8")
    _ollama_call_count = 0
    _logger.info(f"run_ollama_text: writing full Ollama exchanges to {_ollama_text_path}")
    _logger.info(f"run_ollama_text: mirroring latest run to {_ollama_mirror_path}")
    return _ollama_text_path
