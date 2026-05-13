from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest

import bot.main as bot_main


def test_main_exits_on_settings_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_settings_error() -> NoReturn:
        raise RuntimeError("missing token")

    monkeypatch.setattr(bot_main, "load_dotenv", lambda _path: None)
    monkeypatch.setattr(bot_main.Settings, "from_env", staticmethod(raise_settings_error))

    with pytest.raises(SystemExit) as exc_info:
        bot_main.main()

    assert str(exc_info.value) == "Configuration error: missing token"


def test_main_exits_on_debug_logger_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_logger_error(*, level_name: str, log_path: Path) -> NoReturn:
        _ = (level_name, log_path)
        raise RuntimeError("bad debug level")

    settings = SimpleNamespace(
        debug=SimpleNamespace(level_name="LOUD"),
        storage=SimpleNamespace(debug_log_path=Path("cdbot.log")),
    )
    monkeypatch.setattr(bot_main, "load_dotenv", lambda _path: None)
    monkeypatch.setattr(bot_main.Settings, "from_env", staticmethod(lambda: settings))
    monkeypatch.setattr(bot_main, "configure_debug_logger", raise_logger_error)

    with pytest.raises(SystemExit) as exc_info:
        bot_main.main()

    assert str(exc_info.value) == "Configuration error: bad debug level"
