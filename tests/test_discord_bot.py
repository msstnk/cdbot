"""Tests for Discord bot helpers and command handling."""
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access,too-few-public-methods

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import discord
import pytest

from bot.codex_runner import TurnResult
from bot.config import CodexSettings, DebugSettings, LocaleSettings, Settings, StorageSettings
from bot.conversation_state import ActiveTurn
from bot.debug_logging import configure_debug_logger
from bot.discord_bot import CodexDiscordBot
from bot.localization import DEFAULT_LOCALE, DEFAULT_LOCALES_PATH, Messages


def test_resolve_cwd_accepts_workspace_descendant(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    target = workspace / "src"
    target.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace))

    resolved = bot._resolve_cwd("src")

    assert resolved == str(target.resolve())


def test_resolve_cwd_rejects_parent_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    bot = CodexDiscordBot(_settings(tmp_path, workspace))

    resolved = bot._resolve_cwd("../outside")

    assert resolved is None


def test_resolve_cwd_rejects_absolute_path_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    bot = CodexDiscordBot(_settings(tmp_path, workspace))

    resolved = bot._resolve_cwd(str(outside.resolve()))

    assert resolved is None


class DummyAuthor:
    def __init__(self) -> None:
        self.bot = False
        self.id = 1


class DummyChannel:
    def __init__(self) -> None:
        self.id = 123


class DummyMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.author = DummyAuthor()
        self.channel = DummyChannel()
        self.guild = None
        self.replies: list[str] = []

    async def reply(self, content: str, *, mention_author: bool = False) -> None:
        assert mention_author is False
        self.replies.append(content)


class FakeRunner:
    def __init__(self, before_return: Callable[[], Any] | None = None) -> None:
        self.before_return = before_return

    async def run_turn(self, request: object, active_turn: object, on_block: object) -> TurnResult:
        _ = (request, active_turn, on_block)
        if self.before_return is not None:
            self.before_return()
        return TurnResult(
            thread_id="thread-new",
            turn_id="turn-1",
            status="completed",
            assistant_text="",
            resumed=False,
        )


def _change_model_and_clear_session(
    bot: CodexDiscordBot, conversation_key: str, model: str
) -> None:
    bot._session_store.set_model(conversation_key, model)
    bot._session_store.clear(conversation_key)


@pytest.mark.asyncio
async def test_handle_clear_replies_with_effective_model_and_cwd(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace))
    conversation_key = "dm:123"
    bot._session_store.save_session(
        conversation_key,
        session_id="thread-1",
        session_model="gpt-5.4",
        session_cwd=str(workspace),
    )
    bot._session_store.set_model(conversation_key, "gpt-5.5")
    bot._session_store.set_cwd(conversation_key, str((workspace / "src").resolve()))
    message = DummyMessage("/clear")

    await bot._handle_clear(cast(discord.Message, message), conversation_key)

    assert message.replies == [
        _messages().format(
            "discord.session_cleared",
            model="gpt-5.5",
            cwd=str((workspace / "src").resolve()),
        )
    ]
    assert bot._session_store.get_session_id(conversation_key) == ""
    assert bot._session_store.get_model(conversation_key, "fallback") == "gpt-5.5"
    assert bot._session_store.get_cwd(conversation_key, "fallback") == str(
        (workspace / "src").resolve()
    )


@pytest.mark.asyncio
async def test_handle_model_clears_saved_session_and_replies_with_notice(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace))
    conversation_key = "dm:123"
    bot._session_store.save_session(
        conversation_key,
        session_id="thread-1",
        session_model="gpt-5.4",
        session_cwd=str(workspace),
    )
    message = DummyMessage("/model gpt-5.5")

    await bot._handle_model(cast(discord.Message, message), conversation_key)

    assert message.replies == [
        _messages().format("discord.model_updated_with_clear", model="gpt-5.5")
    ]
    assert bot._session_store.get_session_id(conversation_key) == ""
    assert bot._session_store.get_model(conversation_key, "fallback") == "gpt-5.5"


@pytest.mark.asyncio
async def test_on_message_persists_discord_user_id(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace))
    message = DummyMessage("/model")

    await bot.on_message(cast(discord.Message, message))

    assert bot._session_store.get_discord_user_id("dm:123") == 1


@pytest.mark.asyncio
async def test_on_message_rejects_non_whitelisted_user(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace, whitelisted_users={999}))
    message = DummyMessage("/model")

    await bot.on_message(cast(discord.Message, message))

    assert message.replies == [_messages().text("discord.user_not_allowed")]
    assert bot._session_store.get_discord_user_id("dm:123") == 0


@pytest.mark.asyncio
async def test_on_message_logs_dm_before_user_check(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    log_path = tmp_path / "cdbot.log"
    configure_debug_logger(level_name="DEBUG", log_path=log_path)
    bot = CodexDiscordBot(_settings(tmp_path, workspace))
    message = DummyMessage("/model")

    await bot.on_message(cast(discord.Message, message))

    assert "DEBUG cdbot discord.dm.received message='/model' dm_id=123 user_id=1" in (
        log_path.read_text(encoding="utf-8")
    )


@pytest.mark.asyncio
async def test_on_message_logs_rejection(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    log_path = tmp_path / "cdbot.log"
    configure_debug_logger(level_name="INFO", log_path=log_path)
    bot = CodexDiscordBot(_settings(tmp_path, workspace, whitelisted_users={999}))
    message = DummyMessage("/model")

    await bot.on_message(cast(discord.Message, message))

    assert "INFO cdbot discord.dm.rejected dm_id=123 user_id=1 reason=not_whitelisted" in (
        log_path.read_text(encoding="utf-8")
    )


@pytest.mark.asyncio
async def test_run_turn_does_not_restore_old_session_after_model_change(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace))
    conversation_key = "dm:123"
    bot._session_store.set_model(conversation_key, "gpt-5.4")
    bot._session_store.save_session(
        conversation_key,
        session_id="thread-old",
        session_model="gpt-5.4",
        session_cwd=str(workspace),
    )
    bot._runner = cast(
        Any,
        FakeRunner(
            before_return=lambda: _change_model_and_clear_session(
                bot, conversation_key, "gpt-5.5"
            )
        ),
    )
    active_turn = ActiveTurn(conversation_key=conversation_key)
    message = DummyMessage("hello")

    await bot._run_turn(cast(discord.Message, message), active_turn, "hello")

    assert bot._session_store.get_session_id(conversation_key) == ""
    assert bot._session_store.get_model(conversation_key, "fallback") == "gpt-5.5"


def _settings(
    tmp_path: Path,
    workspace: Path,
    *,
    whitelisted_users: set[int] | frozenset[int] = frozenset(),
) -> Settings:
    codex_bin = tmp_path / "codex"
    codex_bin.write_text("", encoding="utf-8")
    return Settings(
        discord_bot_token="token",
        whitelisted_users=frozenset(whitelisted_users),
        codex=CodexSettings(
            bin_path=str(codex_bin),
            home_path=tmp_path / ".codex",
            default_model="gpt-5.5",
            workspace_cwd=str(workspace.resolve()),
        ),
        approval_timeout_sec=60.0,
        storage=StorageSettings(
            session_store_path=tmp_path / "sessions.jsonl",
            debug_log_path=tmp_path / "cdbot.log",
        ),
        debug=DebugSettings(level_name="OFF"),
        localization=LocaleSettings(
            locale=DEFAULT_LOCALE,
            path=DEFAULT_LOCALES_PATH.resolve(),
        ),
    )


def _messages() -> Messages:
    return Messages.load(DEFAULT_LOCALE, DEFAULT_LOCALES_PATH.resolve())
