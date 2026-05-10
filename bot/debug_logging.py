"""Debug logging utilities for Codex SDK traffic."""

from __future__ import annotations

import json
import logging
from logging import Logger
from pathlib import Path
from typing import Any

TRACE_LEVEL = 5
LOGGER_NAME = "cdbot"

logging.addLevelName(TRACE_LEVEL, "TRACE")


def configure_debug_logger(*, level_name: str, log_path: Path) -> Logger:
    """Configure and return the project debug logger."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(_resolve_log_level(level_name))

    if logger.level > logging.CRITICAL:
        return logger

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logger.level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    return logger


def get_logger() -> Logger:
    """Return the project debug logger."""
    return logging.getLogger(LOGGER_NAME)


def trace(logger: Logger, message: str, *args: object) -> None:
    """Log a message at the custom TRACE level."""
    logger.log(TRACE_LEVEL, message, *args)


def dump_for_log(value: Any) -> str:
    """Serialize an SDK object into stable JSON for log output."""
    serialized = _serialize(value)
    return json.dumps(serialized, ensure_ascii=False, sort_keys=True)


def _resolve_log_level(level_name: str) -> int:
    normalized = level_name.strip().upper()
    if normalized in {"OFF", "NONE"}:
        return logging.CRITICAL + 100
    if normalized == "TRACE":
        return TRACE_LEVEL
    if normalized == "DEBUG":
        return logging.DEBUG
    if normalized == "INFO":
        return logging.INFO
    if normalized == "WARNING":
        return logging.WARNING
    if normalized == "ERROR":
        return logging.ERROR
    raise RuntimeError(
        "CDBOT_DEBUG_LEVEL must be one of OFF, ERROR, WARNING, INFO, DEBUG, TRACE"
    )


def _serialize(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _serialize(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if hasattr(value, "__dict__"):
        result: dict[str, Any] = {}
        for name, attr in vars(value).items():
            if name.startswith("_"):
                continue
            result[name] = _serialize(attr)
        if result:
            return result
    return repr(value)
