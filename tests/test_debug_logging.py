from __future__ import annotations

from pathlib import Path

import pytest

from bot.debug_logging import TRACE_LEVEL, configure_debug_logger


def test_configure_debug_logger_creates_log_file(tmp_path: Path) -> None:
    log_path = tmp_path / "cdbot.log"

    logger = configure_debug_logger(level_name="DEBUG", log_path=log_path)
    logger.debug("hello")

    assert log_path.exists()
    assert "hello" in log_path.read_text(encoding="utf-8")


def test_configure_debug_logger_supports_trace_level(tmp_path: Path) -> None:
    log_path = tmp_path / "cdbot.log"

    logger = configure_debug_logger(level_name="TRACE", log_path=log_path)

    assert logger.level == TRACE_LEVEL


def test_configure_debug_logger_rejects_unknown_level(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        configure_debug_logger(level_name="LOUD", log_path=tmp_path / "cdbot.log")
