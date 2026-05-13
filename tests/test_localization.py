"""Tests for localization loading and environment-backed settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from bot.config import Settings
from bot.localization import Messages


def test_messages_load_requested_locale() -> None:
    """Load a known locale and confirm a representative key exists."""
    messages = Messages.load("en_US")

    assert messages.locale == "en_US"
    assert messages.text("discord.runner_not_ready")


def test_settings_from_env_reads_locale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Settings should pick up the requested locale from the environment."""
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
    """Settings should parse comma-separated whitelisted Discord user ids."""
    codex_bin = tmp_path / "codex"
    codex_bin.write_text("", encoding="utf-8")

    monkeypatch.setenv("CDBOT_DISCORD_BOT_TOKEN", "token")
    monkeypatch.setenv("CDBOT_CODEX_BIN", str(codex_bin))
    monkeypatch.setenv("CDBOT_WHITELISTED_USERS", "194049781591572480, 194049781591572481")

    settings = Settings.from_env()

    assert settings.discord.whitelisted_users == {
        194049781591572480,
        194049781591572481,
    }


def test_settings_from_env_reads_voice_control_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Settings should parse the voice-control feature flag from the environment."""
    codex_bin = tmp_path / "codex"
    codex_bin.write_text("", encoding="utf-8")

    monkeypatch.setenv("CDBOT_DISCORD_BOT_TOKEN", "token")
    monkeypatch.setenv("CDBOT_CODEX_BIN", str(codex_bin))
    monkeypatch.setenv("CDBOT_ENABLE_VOICE_CONTROL", "true")

    settings = Settings.from_env()

    assert settings.openai.enable_voice_control is True


def test_settings_from_env_reads_integer_approval_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Settings should parse approval timeout as whole seconds."""
    codex_bin = tmp_path / "codex"
    codex_bin.write_text("", encoding="utf-8")

    monkeypatch.setenv("CDBOT_DISCORD_BOT_TOKEN", "token")
    monkeypatch.setenv("CDBOT_CODEX_BIN", str(codex_bin))
    monkeypatch.setenv("CDBOT_APPROVAL_TIMEOUT_SEC", "5")

    settings = Settings.from_env()

    assert settings.discord.approval_timeout_sec == 5


def test_settings_from_env_rejects_decimal_approval_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Settings should reject fractional approval timeout values."""
    codex_bin = tmp_path / "codex"
    codex_bin.write_text("", encoding="utf-8")

    monkeypatch.setenv("CDBOT_DISCORD_BOT_TOKEN", "token")
    monkeypatch.setenv("CDBOT_CODEX_BIN", str(codex_bin))
    monkeypatch.setenv("CDBOT_APPROVAL_TIMEOUT_SEC", "1.5")

    expected = "CDBOT_APPROVAL_TIMEOUT_SEC must be an integer"
    with pytest.raises(RuntimeError, match=expected):
        Settings.from_env()
