from __future__ import annotations

from pathlib import Path

import pytest

from bot.config import Settings
from bot.localization import Messages


def test_messages_load_requested_locale() -> None:
    messages = Messages.load("en_US")

    assert messages.locale == "en_US"
    assert messages.text("discord.runner_not_ready")


def test_settings_from_env_reads_locale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    codex_bin = tmp_path / "codex"
    codex_bin.write_text("", encoding="utf-8")

    monkeypatch.setenv("CDBOT_DISCORD_BOT_TOKEN", "token")
    monkeypatch.setenv("CDBOT_CODEX_BIN", str(codex_bin))
    monkeypatch.setenv("CDBOT_LOCALE", "en_US")

    settings = Settings.from_env()

    assert settings.localization.locale == "en_US"


def test_settings_from_env_reads_whitelisted_users(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    codex_bin = tmp_path / "codex"
    codex_bin.write_text("", encoding="utf-8")

    monkeypatch.setenv("CDBOT_DISCORD_BOT_TOKEN", "token")
    monkeypatch.setenv("CDBOT_CODEX_BIN", str(codex_bin))
    monkeypatch.setenv("CDBOT_WHITELISTED_USERS", "194049781591572480, 194049781591572481")

    settings = Settings.from_env()

    assert settings.whitelisted_users == {
        194049781591572480,
        194049781591572481,
    }
