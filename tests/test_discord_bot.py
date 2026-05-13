"""Tests for Discord bot helpers and command handling."""
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access,too-few-public-methods

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import discord
import pytest

from bot.codex_runner import AssistantOutputBlock, TurnResult
from bot.config import (
    CodexSettings,
    DebugSettings,
    DiscordSettings,
    LocaleSettings,
    OpenAISettings,
    Settings,
    StorageSettings,
)
from bot.conversation_state import ActiveTurn
from bot.debug_logging import configure_debug_logger
from bot.discord_bot import CodexDiscordBot, DiscordReplyStream, _format_token_usage_text
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


def test_format_token_usage_text_uses_subtext_and_m_suffix_for_six_figures() -> None:
    formatted = _format_token_usage_text(
        _messages(),
        {
            "cached_input_tokens": 1_300_000,
            "input_tokens": 1_400_000,
            "output_tokens": 10_500,
            "total_tokens": 1_410_500,
        },
    )

    assert (
        formatted
        == "-# *Token usage for this session - 0.1M uncached + 1.3M cached input, 10.5k output*"
    )


class DummyAuthor:
    def __init__(self) -> None:
        self.bot = False
        self.id = 1


class DummyChannel:
    def __init__(self) -> None:
        self.id = 123
        self.sent_messages: list[Any] = []

    async def send(self, content: str) -> Any:
        self.sent_messages.append(content)
        return content


class DummyTurnChannel(DummyChannel):
    def __init__(self) -> None:
        super().__init__()
        self.sent_messages: list[DummySentMessage] = []

    async def send(self, content: str) -> DummySentMessage:
        sent = DummySentMessage(content)
        self.sent_messages.append(sent)
        return sent


class DummyVoiceAttachment:
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        filename: str = "voice.ogg",
        content_type: str = "audio/ogg",
        payload: bytes = b"voice",
        attachment_id: int = 999,
        size: int | None = None,
    ) -> None:
        self.filename = filename
        self.content_type = content_type
        self.id = attachment_id
        self.size = len(payload) if size is None else size
        self._payload = payload

    def is_voice_message(self) -> bool:
        return True

    async def read(self) -> bytes:
        return self._payload


class DummyMessage:
    def __init__(self, content: str, *, attachments: list[Any] | None = None) -> None:
        self.content = content
        self.author = DummyAuthor()
        self.channel = DummyChannel()
        self.guild = None
        self.attachments = attachments or []
        self.replies: list[str] = []

    async def reply(self, content: str, *, mention_author: bool = False) -> DummySentMessage:
        assert mention_author is False
        self.replies.append(content)
        return DummySentMessage(content)


class DummySentMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.edits: list[str] = []

    async def edit(self, *, content: str) -> None:
        self.content = content
        self.edits.append(content)


class DummyStreamChannel:
    def __init__(self) -> None:
        self.sent_messages: list[DummySentMessage] = []

    async def send(self, content: str) -> DummySentMessage:
        sent = DummySentMessage(content)
        self.sent_messages.append(sent)
        return sent


class DummyStreamSourceMessage:
    def __init__(self) -> None:
        self.channel = DummyStreamChannel()
        self.replies: list[DummySentMessage] = []

    async def reply(self, content: str, *, mention_author: bool = False) -> DummySentMessage:
        assert mention_author is False
        sent = DummySentMessage(content)
        self.replies.append(sent)
        return sent


class DummyTurnMessage(DummyMessage):
    def __init__(self, content: str) -> None:
        super().__init__(content)
        self.channel = DummyTurnChannel()
        self.reply_messages: list[DummySentMessage] = []

    async def reply(
        self, content: str, *, mention_author: bool = False
    ) -> DummySentMessage:
        assert mention_author is False
        self.replies.append(content)
        sent = DummySentMessage(content)
        self.reply_messages.append(sent)
        return sent


class FakeRunner:
    def __init__(
        self,
        before_return: Callable[[], Any] | None = None,
        *,
        assistant_text: str = "",
        final_assistant_text: str | None = None,
        token_usage_last: dict[str, int] | None = None,
    ) -> None:
        self.before_return = before_return
        self.assistant_text = assistant_text
        self.final_assistant_text = (
            assistant_text if final_assistant_text is None else final_assistant_text
        )
        self.token_usage_last = token_usage_last or {}

    async def run_turn(self, request: object, active_turn: object, on_block: object) -> TurnResult:
        _ = (request, active_turn, on_block)
        if self.before_return is not None:
            self.before_return()
        return TurnResult(
            thread_id="thread-new",
            turn_id="turn-1",
            status="completed",
            assistant_text=self.assistant_text,
            final_assistant_text=self.final_assistant_text,
            resumed=False,
            token_usage_last=dict(self.token_usage_last),
        )


class FakeCapturingRunner:
    def __init__(self) -> None:
        self.request: object | None = None

    async def run_turn(self, request: object, active_turn: object, on_block: object) -> TurnResult:
        _ = (active_turn, on_block)
        self.request = request
        return TurnResult(
            thread_id="thread-new",
            turn_id="turn-1",
            status="completed",
            assistant_text="",
            final_assistant_text="",
            resumed=False,
            token_usage_last={},
        )


class FakeStreamingRunner:
    async def run_turn(self, request: object, active_turn: object, on_block: object) -> TurnResult:
        _ = (request, active_turn)
        await cast(Any, on_block)(AssistantOutputBlock(text="first"))
        await cast(Any, on_block)(AssistantOutputBlock(text="second"))
        return TurnResult(
            thread_id="thread-new",
            turn_id="turn-1",
            status="completed",
            assistant_text="firstsecondfinal",
            final_assistant_text="final",
            resumed=False,
            token_usage_last={
                "cached_input_tokens": 0,
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_tokens": 1200,
            },
        )


class FakeVoiceTranscriber:
    def __init__(
        self,
        transcript: str = "hello from voice",
        error: Exception | None = None,
    ) -> None:
        self._error = error
        self._transcript = transcript
        self.attachments: list[Any] = []

    async def __call__(self, attachment: object) -> str:
        self.attachments.append(attachment)
        if self._error is not None:
            raise self._error
        return self._transcript


def _change_model_and_clear_session(
    bot: CodexDiscordBot, conversation_key: str, model: str
) -> None:
    bot._session_store.set_model(conversation_key, model)
    bot._session_store.clear(conversation_key)


@pytest.mark.asyncio
async def test_reply_stream_sends_all_chunks_to_channel() -> None:
    source = DummyStreamSourceMessage()
    stream = DiscordReplyStream(cast(discord.Message, source), _messages())

    await stream.add_block(AssistantOutputBlock(text="a" * 2001))

    assert not source.replies
    assert [message.content for message in source.channel.sent_messages] == ["a" * 2000, "a"]


@pytest.mark.asyncio
async def test_reply_stream_fail_edits_existing_first_message() -> None:
    source = DummyStreamSourceMessage()
    stream = DiscordReplyStream(cast(discord.Message, source), _messages())
    await stream.add_block(AssistantOutputBlock(text="partial"))

    await stream.fail("boom")

    expected = _messages().format("discord.error", error_text="boom")
    assert source.channel.sent_messages[0].content == expected
    assert source.channel.sent_messages[0].edits == [expected]
    assert len(source.channel.sent_messages) == 1


@pytest.mark.asyncio
async def test_reply_stream_finalize_reports_non_success_without_output() -> None:
    source = DummyStreamSourceMessage()
    stream = DiscordReplyStream(cast(discord.Message, source), _messages())

    await stream.finalize("", status="failed")

    status_text = _messages().format("discord.turn_finished_with_status", status="failed")
    assert [message.content for message in source.channel.sent_messages] == [
        _messages().format("discord.error", error_text=status_text)
    ]


@pytest.mark.asyncio
async def test_reply_stream_append_tail_edits_last_message_when_it_fits() -> None:
    source = DummyStreamSourceMessage()
    stream = DiscordReplyStream(cast(discord.Message, source), _messages())
    await stream.add_block(AssistantOutputBlock(text="assistant reply"))

    await stream.add_block(AssistantOutputBlock(text="tail"))

    assert [message.content for message in source.channel.sent_messages] == [
        "assistant reply\n\ntail"
    ]
    assert source.channel.sent_messages[0].edits == ["assistant reply\n\ntail"]


@pytest.mark.asyncio
async def test_reply_stream_append_tail_sends_new_message_when_last_chunk_is_full() -> None:
    source = DummyStreamSourceMessage()
    stream = DiscordReplyStream(cast(discord.Message, source), _messages())
    await stream.add_block(AssistantOutputBlock(text="a" * 2000))

    await stream.add_block(AssistantOutputBlock(text="tail"))

    assert [message.content for message in source.channel.sent_messages] == ["a" * 2000, "tail"]


@pytest.mark.asyncio
async def test_reply_stream_starts_new_message_after_approval_boundary() -> None:
    source = DummyStreamSourceMessage()
    stream = DiscordReplyStream(cast(discord.Message, source), _messages())

    await stream.add_block(AssistantOutputBlock(text="before approval"))
    await stream.add_block(
        AssistantOutputBlock(text="after approval", starts_new_message=True)
    )

    assert [message.content for message in source.channel.sent_messages] == [
        "before approval",
        "after approval",
    ]


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

    assert message.channel.sent_messages == [
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

    assert message.channel.sent_messages == [
        _messages().format("discord.model_updated_with_clear", model="gpt-5.5")
    ]
    assert bot._session_store.get_session_id(conversation_key) == ""
    assert bot._session_store.get_model(conversation_key, "fallback") == "gpt-5.5"


@pytest.mark.asyncio
async def test_handle_model_without_argument_replies_with_current_model(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace))
    conversation_key = "dm:123"
    bot._session_store.set_model(conversation_key, "gpt-5.4")
    message = DummyMessage("/model")

    await bot._handle_model(cast(discord.Message, message), conversation_key)

    assert message.channel.sent_messages == [
        _messages().format("discord.current_model", current="gpt-5.4")
    ]


@pytest.mark.asyncio
async def test_handle_cwd_without_argument_replies_with_current_cwd(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    target = workspace / "src"
    target.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace))
    conversation_key = "dm:123"
    bot._session_store.set_cwd(conversation_key, str(target.resolve()))
    message = DummyMessage("/cwd")

    await bot._handle_cwd(cast(discord.Message, message), conversation_key)

    assert message.channel.sent_messages == [
        _messages().format("discord.current_cwd", current=str(target.resolve()))
    ]


@pytest.mark.asyncio
async def test_handle_cwd_rejects_invalid_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace))
    conversation_key = "dm:123"
    message = DummyMessage("/cwd missing")

    await bot._handle_cwd(cast(discord.Message, message), conversation_key)

    assert message.channel.sent_messages == [_messages().text("discord.invalid_cwd")]
    assert bot._session_store.get_cwd(conversation_key, "fallback") == "fallback"


@pytest.mark.asyncio
async def test_handle_cwd_updates_to_workspace_descendant(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    target = workspace / "src"
    target.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace))
    conversation_key = "dm:123"
    message = DummyMessage("/cwd src")

    await bot._handle_cwd(cast(discord.Message, message), conversation_key)

    assert message.channel.sent_messages == [
        _messages().format("discord.cwd_updated", cwd=str(target.resolve()))
    ]
    assert bot._session_store.get_cwd(conversation_key, "fallback") == str(target.resolve())


@pytest.mark.asyncio
async def test_on_message_does_not_treat_command_prefix_as_command(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace))
    runner = FakeCapturingRunner()
    bot._runner = cast(Any, runner)
    message = DummyMessage("/modelx")

    await bot.on_message(cast(discord.Message, message))
    active_turn = await bot._conversation_state.get("dm:123")
    assert active_turn is not None and active_turn.task is not None
    await active_turn.task

    request = cast(Any, runner.request)
    assert request.prompt == "/modelx"
    assert not message.replies


@pytest.mark.asyncio
async def test_on_message_persists_discord_user_id(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace))
    message = DummyMessage("/model")

    await bot.on_message(cast(discord.Message, message))

    assert bot._session_store.get_discord_user_id("dm:123") == 1


@pytest.mark.asyncio
async def test_on_message_uses_voice_message_transcript(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace, enable_voice_control=True))
    bot._voice_transcriber = cast(Any, FakeVoiceTranscriber("transcribed voice"))
    runner = FakeCapturingRunner()
    bot._runner = cast(Any, runner)
    message = DummyMessage("", attachments=[DummyVoiceAttachment()])

    await bot.on_message(cast(discord.Message, message))
    active_turn = await bot._conversation_state.get("dm:123")
    assert active_turn is not None and active_turn.task is not None
    await active_turn.task

    request = cast(Any, runner.request)
    assert request.prompt == "[Voice message transcript]\ntranscribed voice"
    assert not message.replies


@pytest.mark.asyncio
async def test_on_message_combines_voice_transcript_and_text(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace, enable_voice_control=True))
    bot._voice_transcriber = cast(Any, FakeVoiceTranscriber("transcribed voice"))
    voice_attachment = DummyVoiceAttachment()

    prompt = await bot._build_prompt(
        cast(discord.Message, DummyMessage("please summarize", attachments=[voice_attachment])),
        cast(discord.Attachment, voice_attachment),
    )

    assert prompt == (
        "[Voice message transcript]\ntranscribed voice\n\n[User text]\nplease summarize"
    )


@pytest.mark.asyncio
async def test_on_message_replies_when_voice_control_is_disabled(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace))
    message = DummyMessage("", attachments=[DummyVoiceAttachment()])

    await bot.on_message(cast(discord.Message, message))

    assert message.channel.sent_messages == [_messages().text("discord.voice_control_disabled")]


@pytest.mark.asyncio
async def test_on_message_replies_when_voice_transcription_is_unconfigured(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace, enable_voice_control=True))
    message = DummyMessage("", attachments=[DummyVoiceAttachment()])

    await bot.on_message(cast(discord.Message, message))

    assert message.channel.sent_messages == [_messages().text("discord.voice_not_configured")]


@pytest.mark.asyncio
async def test_on_message_replies_when_voice_transcription_fails(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace, enable_voice_control=True))
    bot._voice_transcriber = cast(Any, FakeVoiceTranscriber(error=RuntimeError("boom")))
    message = DummyMessage("", attachments=[DummyVoiceAttachment()])

    await bot.on_message(cast(discord.Message, message))

    assert message.channel.sent_messages == [
        _messages().format("discord.voice_transcription_failed", error_text="boom")
    ]


@pytest.mark.asyncio
async def test_on_message_rejects_non_whitelisted_user(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace, whitelisted_users={999}))
    message = DummyMessage("/model")

    await bot.on_message(cast(discord.Message, message))

    assert message.channel.sent_messages == [
        _messages().format("discord.user_not_allowed", user_id=1)
    ]
    assert bot._session_store.get_discord_user_id("dm:123") == 0


@pytest.mark.asyncio
async def test_on_message_rejects_non_whitelisted_voice_user_before_transcription(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(
        _settings(tmp_path, workspace, whitelisted_users={999}, enable_voice_control=True)
    )
    transcriber = FakeVoiceTranscriber("transcribed voice")
    bot._voice_transcriber = cast(Any, transcriber)
    message = DummyMessage("", attachments=[DummyVoiceAttachment()])

    await bot.on_message(cast(discord.Message, message))

    assert message.channel.sent_messages == [
        _messages().format("discord.user_not_allowed", user_id=1)
    ]
    assert not transcriber.attachments


@pytest.mark.asyncio
async def test_on_message_does_not_execute_clear_command_for_voice_message(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace, enable_voice_control=True))
    bot._voice_transcriber = cast(Any, FakeVoiceTranscriber("transcribed voice"))
    runner = FakeCapturingRunner()
    bot._runner = cast(Any, runner)
    conversation_key = "dm:123"
    bot._session_store.save_session(
        conversation_key,
        session_id="thread-1",
        session_model="gpt-5.5",
        session_cwd=str(workspace),
    )
    message = DummyMessage("/clear", attachments=[DummyVoiceAttachment()])

    await bot.on_message(cast(discord.Message, message))
    active_turn = await bot._conversation_state.get(conversation_key)
    assert active_turn is not None and active_turn.task is not None
    await active_turn.task

    request = cast(Any, runner.request)
    assert request.prompt == "[Voice message transcript]\ntranscribed voice\n\n[User text]\n/clear"
    assert bot._session_store.get_session_id(conversation_key) == "thread-new"
    assert not message.replies


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


@pytest.mark.asyncio
async def test_run_turn_appends_token_usage_message(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace))
    bot._runner = cast(
        Any,
        FakeRunner(
            assistant_text="assistant reply",
            token_usage_last={
                "cached_input_tokens": 8000,
                "input_tokens": 12000,
                "output_tokens": 5000,
                "total_tokens": 17000,
            },
        ),
    )
    active_turn = ActiveTurn(conversation_key="dm:123")
    message = DummyTurnMessage("hello")

    await bot._run_turn(cast(discord.Message, message), active_turn, "hello")

    assert [sent.content for sent in message.channel.sent_messages] == [
        "assistant reply\n\n"
        "-# *Token usage for this session - "
        "4.0k uncached + 8.0k cached input, 5.0k output*"
    ]


@pytest.mark.asyncio
async def test_run_turn_sends_intermediate_and_final_messages_separately(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    bot = CodexDiscordBot(_settings(tmp_path, workspace))
    bot._runner = cast(Any, FakeStreamingRunner())
    active_turn = ActiveTurn(conversation_key="dm:123")
    message = DummyTurnMessage("hello")

    await bot._run_turn(cast(discord.Message, message), active_turn, "hello")

    assert [sent.content for sent in message.channel.sent_messages] == [
        "first\n\nsecond",
        "final\n\n-# *Token usage for this session - 1.0k uncached + 0 cached input, 200 output*",
    ]


def _settings(
    tmp_path: Path,
    workspace: Path,
    *,
    whitelisted_users: set[int] | frozenset[int] = frozenset(),
    enable_voice_control: bool = False,
) -> Settings:
    codex_bin = tmp_path / "codex"
    codex_bin.write_text("", encoding="utf-8")
    return Settings(
        discord=DiscordSettings(
            bot_token="token",
            whitelisted_users=frozenset(whitelisted_users),
            approval_timeout_sec=60,
        ),
        codex=CodexSettings(
            bin_path=str(codex_bin),
            home_path=tmp_path / ".codex",
            default_model="gpt-5.5",
            workspace_cwd=str(workspace.resolve()),
        ),
        openai=OpenAISettings(
            enable_voice_control=enable_voice_control,
            api_key="",
            transcription_model="whisper-1",
        ),
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
