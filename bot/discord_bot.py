"""Discord client implementation for Codex DM conversations."""

from __future__ import annotations

import asyncio
from pathlib import Path

import discord

from .approval_router import ApprovalRouter
from .audio_transcriber import VoiceTranscriber, make_openai_voice_transcriber
from .codex_runner import (
    AssistantOutputBlock,
    CodexRunner,
    RunnerDependencies,
    RunnerEnvironment,
    SessionContext,
    TurnContext,
    TurnRequest,
)
from .config import DISCORD_MESSAGE_LIMIT, Settings
from .conversation_state import ActiveTurn, ConversationState
from .debug_logging import get_logger
from .localization import Messages
from .session_store import SessionStore
from .text import split_discord_text


class DiscordReplyStream:
    """Append Codex output blocks to the source DM channel."""

    def __init__(self, source_message: discord.Message, messages: Messages) -> None:
        self._source_message = source_message
        self._messages_catalog = messages
        self._messages: list[discord.Message] = []
        self._lock = asyncio.Lock()

    async def add_block(self, block: AssistantOutputBlock) -> None:
        """Render one non-final assistant block, editing the last message when possible."""
        chunks = split_discord_text(block.text, DISCORD_MESSAGE_LIMIT)
        if not chunks:
            return
        async with self._lock:
            if not self._messages or block.starts_new_message:
                await self._send_chunks(chunks)
                return

            last_message = self._messages[-1]
            separator = "\n\n" if last_message.content else ""
            candidate = f"{last_message.content}{separator}{chunks[0]}"
            start_index = 0
            if len(candidate) <= DISCORD_MESSAGE_LIMIT:
                await last_message.edit(content=candidate)
                start_index = 1

            await self._send_chunks(chunks[start_index:])

    async def finalize(
        self,
        final_text: str,
        *,
        status: str,
        token_usage_text: str = "",
    ) -> None:
        """Send the final assistant response as a fresh message."""
        if final_text:
            rendered = final_text
            if token_usage_text:
                rendered = f"{rendered}\n\n{token_usage_text}"
            chunks = split_discord_text(rendered, DISCORD_MESSAGE_LIMIT)
            if chunks:
                async with self._lock:
                    await self._send_chunks(chunks)
            return

        if not self._messages and status not in {"completed", "succeeded"}:
            await self.fail(
                self._messages_catalog.format(
                    "discord.turn_finished_with_status", status=status
                )
            )

    async def _send_chunks(self, chunks: list[str]) -> None:
        """Send already-split chunks as new Discord messages."""
        for chunk in chunks:
            sent = await self._source_message.channel.send(chunk)
            self._messages.append(sent)

    async def fail(self, error_text: str) -> None:
        """Render an error in the reply stream."""
        text = self._messages_catalog.format("discord.error", error_text=error_text)
        chunks = split_discord_text(text, DISCORD_MESSAGE_LIMIT)
        if not chunks:
            return
        if not self._messages:
            first = await self._source_message.channel.send(chunks[0])
            self._messages.append(first)
            start_index = 1
        else:
            await self._messages[0].edit(content=chunks[0])
            start_index = 1
        for chunk in chunks[start_index:]:
            sent = await self._source_message.channel.send(chunk)
            self._messages.append(sent)


class CodexDiscordBot(discord.Client):
    """Discord DM client that forwards prompts to a Codex runner."""

    def __init__(self, settings: Settings) -> None:
        intents = discord.Intents.none()
        intents.dm_messages = True
        super().__init__(intents=intents)
        self._settings = settings
        self._messages = Messages.load(
            settings.localization.locale,
            settings.localization.path,
        )
        self._session_store = SessionStore(settings.storage.session_store_path)
        self._conversation_state = ConversationState()
        self._approval_router = ApprovalRouter(
            timeout_sec=settings.discord.approval_timeout_sec,
            messages=self._messages,
        )
        self._runner: CodexRunner | None = None
        self._voice_transcriber: VoiceTranscriber | None = None
        if settings.openai.enable_voice_control and settings.openai.api_key:
            self._voice_transcriber = make_openai_voice_transcriber(
                api_key=settings.openai.api_key,
                model=settings.openai.transcription_model,
            )

    async def setup_hook(self) -> None:
        """Create the Codex runner once Discord has an event loop."""
        self._runner = CodexRunner(
            environment=RunnerEnvironment(
                codex_bin=self._settings.codex.bin_path,
                codex_home=self._settings.codex.home_path,
                workspace_cwd=self._settings.codex.workspace_cwd,
                debug_level_name=self._settings.debug.level_name,
            ),
            dependencies=RunnerDependencies(
                approval_router=self._approval_router,
                loop=asyncio.get_running_loop(),
                messages=self._messages,
            ),
        )

    async def on_ready(self) -> None:
        """Log the bot identity after Discord connects."""
        user = self.user
        if user is None:
            return
        get_logger().info(
            "discord.connected user=%s user_id=%s",
            user,
            user.id,
        )

    async def on_message(self, message: discord.Message) -> None:
        """Handle incoming DM commands and prompts."""
        if message.author.bot or message.guild is not None:
            return

        get_logger().debug(
            "discord.dm.received message=%r dm_id=%s user_id=%s attachments=%s",
            message.content,
            message.channel.id,
            message.author.id,
            len(message.attachments),
        )
        content = message.content.strip()
        conversation_key = f"dm:{message.channel.id}"
        if not self._is_whitelisted_user(message.author.id):
            get_logger().info(
                "discord.dm.rejected dm_id=%s user_id=%s reason=not_whitelisted",
                message.channel.id,
                message.author.id,
            )
            await message.channel.send(
                self._messages.format(
                    "discord.user_not_allowed",
                    user_id=message.author.id,
                ),
            )
        else:
            voice_attachment = self._voice_attachment(message.attachments)
            prompt = await self._build_prompt(message, voice_attachment)
            if prompt is None:
                return

            self._session_store.set_discord_user_id(conversation_key, message.author.id)
            if voice_attachment is None and content == "/clear":
                await self._handle_clear(message, conversation_key)
            elif voice_attachment is None and _matches_command(prompt, "/cwd"):
                await self._handle_cwd(message, conversation_key)
            elif voice_attachment is None and _matches_command(prompt, "/model"):
                await self._handle_model(message, conversation_key)
            else:
                await self._handle_prompt(message, conversation_key, prompt)

    async def _handle_clear(
        self,
        message: discord.Message,
        conversation_key: str,
    ) -> None:
        active_turn = await self._conversation_state.clear(conversation_key)
        if active_turn is not None:
            active_turn.discard_output = True
            active_turn.request_interrupt()
        self._session_store.clear(conversation_key)
        model = self._session_store.get_model(
            conversation_key, self._settings.codex.default_model
        )
        cwd = self._session_store.get_cwd(
            conversation_key,
            self._settings.codex.workspace_cwd,
        )
        await message.channel.send(
            self._messages.format("discord.session_cleared", model=model, cwd=cwd)
        )

    async def _handle_model(
        self,
        message: discord.Message,
        conversation_key: str,
    ) -> None:
        _, _, remainder = message.content.partition(" ")
        model = remainder.strip()
        if not model:
            current = self._session_store.get_model(
                conversation_key, self._settings.codex.default_model
            )
            await message.channel.send(
                self._messages.format("discord.current_model", current=current),
            )
            return
        had_saved_session = bool(self._session_store.get_session_id(conversation_key))
        self._session_store.set_model(conversation_key, model)
        self._session_store.clear(conversation_key)
        if had_saved_session:
            reply = self._messages.format("discord.model_updated_with_clear", model=model)
        else:
            reply = self._messages.format("discord.model_updated", model=model)
        await message.channel.send(reply)

    async def _handle_cwd(
        self,
        message: discord.Message,
        conversation_key: str,
    ) -> None:
        _, _, remainder = message.content.partition(" ")
        raw_cwd = remainder.strip()
        if not raw_cwd:
            current = self._session_store.get_cwd(
                conversation_key,
                self._settings.codex.workspace_cwd,
            )
            await message.channel.send(
                self._messages.format("discord.current_cwd", current=current),
            )
            return

        resolved_cwd = self._resolve_cwd(raw_cwd)
        if resolved_cwd is None:
            await message.channel.send(self._messages.text("discord.invalid_cwd"))
            return

        self._session_store.set_cwd(conversation_key, resolved_cwd)
        await message.channel.send(
            self._messages.format("discord.cwd_updated", cwd=resolved_cwd)
        )

    async def _handle_prompt(
        self,
        message: discord.Message,
        conversation_key: str,
        prompt: str,
    ) -> None:
        active_turn = await self._conversation_state.get(conversation_key)
        if active_turn is not None:
            active_turn.queue_steer(prompt)
            await message.channel.send(self._messages.text("discord.turn_input_queued"))
            return

        active_turn = ActiveTurn(conversation_key=conversation_key)
        started = await self._conversation_state.start(active_turn)
        if not started:
            existing = await self._conversation_state.get(conversation_key)
            if existing is not None:
                existing.queue_steer(prompt)
            await message.channel.send(self._messages.text("discord.turn_input_queued"))
            return

        task = asyncio.create_task(self._run_turn(message, active_turn, prompt))
        active_turn.task = task

    async def _run_turn(
        self,
        message: discord.Message,
        active_turn: ActiveTurn,
        prompt: str,
    ) -> None:
        runner = self._runner
        if runner is None:
            await message.channel.send(self._messages.text("discord.runner_not_ready"))
            await self._conversation_state.finish(active_turn)
            return

        conversation_key = active_turn.conversation_key
        reply_stream = DiscordReplyStream(message, self._messages)
        request = self._build_turn_request(message, conversation_key, prompt)
        model = request.session.model
        cwd = request.session.cwd

        try:
            result = await runner.run_turn(request, active_turn, reply_stream.add_block)
            if not active_turn.discard_output:
                current_model = self._session_store.get_model(
                    conversation_key, self._settings.codex.default_model
                )
                if current_model == model:
                    self._session_store.save_session(
                        conversation_key,
                        session_id=result.thread_id,
                        session_model=model,
                        session_cwd=cwd,
                    )
                token_usage_text = _format_token_usage_text(
                    self._messages,
                    result.token_usage_last,
                )
                await reply_stream.finalize(
                    result.final_assistant_text,
                    status=result.status,
                    token_usage_text=token_usage_text,
                )
        except asyncio.CancelledError:
            if not active_turn.discard_output:
                await reply_stream.fail(self._messages.text("discord.turn_cancelled"))

        finally:
            await self._conversation_state.finish(active_turn)

    def _build_turn_request(
        self,
        message: discord.Message,
        conversation_key: str,
        prompt: str,
    ) -> TurnRequest:
        model = self._session_store.get_model(
            conversation_key, self._settings.codex.default_model
        )
        cwd = self._session_store.get_cwd(
            conversation_key, self._settings.codex.workspace_cwd
        )
        session_id = self._session_store.get_session_id(conversation_key)
        session_model = self._session_store.get_session_model(conversation_key)
        session_cwd = self._session_store.get_session_cwd(conversation_key)
        return TurnRequest(
            context=TurnContext(
                conversation_key=conversation_key,
                requester_id=message.author.id,
                channel=message.channel,
            ),
            prompt=prompt,
            session=SessionContext(
                model=model,
                cwd=cwd,
                session_id=session_id,
                session_model=session_model,
                session_cwd=session_cwd,
            ),
        )

    def _is_whitelisted_user(self, user_id: int) -> bool:
        whitelisted_users = self._settings.discord.whitelisted_users
        return not whitelisted_users or user_id in whitelisted_users

    def _resolve_cwd(self, raw_cwd: str) -> str | None:
        workspace_root = Path(self._settings.codex.workspace_cwd).resolve()
        candidate = Path(raw_cwd).expanduser()
        if not candidate.is_absolute():
            candidate = workspace_root / candidate
        resolved = candidate.resolve()
        if not resolved.is_dir():
            return None
        try:
            resolved.relative_to(workspace_root)
        except ValueError:
            return None
        return str(resolved)

    async def _build_prompt(
        self,
        message: discord.Message,
        voice_attachment: discord.Attachment | None,
    ) -> str | None:
        content = message.content.strip()
        if voice_attachment is None:
            return content or None

        transcriber = self._voice_transcriber
        if transcriber is None:
            get_logger().debug(
                "discord.voice.unavailable dm_id=%s user_id=%s enabled=%s",
                message.channel.id,
                message.author.id,
                self._settings.openai.enable_voice_control,
            )
            if self._settings.openai.enable_voice_control:
                await message.channel.send(
                    self._messages.text("discord.voice_not_configured"),
                )
            else:
                await message.channel.send(
                    self._messages.text("discord.voice_control_disabled"),
                )
            return None

        get_logger().debug(
            "discord.voice.detected dm_id=%s user_id=%s attachment_id=%s filename=%s",
            message.channel.id,
            message.author.id,
            voice_attachment.id,
            voice_attachment.filename,
        )
        try:
            transcript = await transcriber(voice_attachment)
        except RuntimeError as exc:
            get_logger().exception(
                "discord.voice.transcription.failed dm_id=%s user_id=%s attachment_id=%s",
                message.channel.id,
                message.author.id,
                voice_attachment.id,
            )
            await message.channel.send(
                self._messages.format(
                    "discord.voice_transcription_failed",
                    error_text=str(exc),
                ),
            )
            return None

        if not transcript:
            get_logger().debug(
                "discord.voice.transcription.empty dm_id=%s user_id=%s attachment_id=%s",
                message.channel.id,
                message.author.id,
                voice_attachment.id,
            )
            await message.channel.send(self._messages.text("discord.voice_transcription_empty"))
            return None

        parts = [f"[Voice message transcript]\n{transcript}"]
        if content:
            parts.append(f"[User text]\n{content}")
        return "\n\n".join(parts)

    @staticmethod
    def _voice_attachment(
        attachments: list[discord.Attachment],
    ) -> discord.Attachment | None:
        for attachment in attachments:
            if attachment.is_voice_message():
                return attachment
        return None


def _matches_command(prompt: str, command: str) -> bool:
    return prompt == command or prompt.startswith(f"{command} ")


def _format_token_usage_text(messages: Messages, token_usage_last: dict[str, int]) -> str:
    if not isinstance(token_usage_last, dict):
        return ""

    def _format_token_count(value: int) -> str:
        if value >= 100_000:
            return f"{value / 1_000_000:.1f}M"
        if value >= 1_000:
            return f"{value / 1_000:.1f}k"
        return str(value)

    cached = int(token_usage_last.get("cached_input_tokens", 0) or 0)
    input_tokens = int(token_usage_last.get("input_tokens", 0) or 0)
    output_tokens = int(token_usage_last.get("output_tokens", 0) or 0)
    total_tokens = int(token_usage_last.get("total_tokens", 0) or 0)
    if cached <= 0 and input_tokens <= 0 and output_tokens <= 0 and total_tokens <= 0:
        return ""

    uncached_input_tokens = max(input_tokens - cached, 0)
    return messages.format(
        "discord.token_usage_last",
        uncached=_format_token_count(uncached_input_tokens if cached > 0 else input_tokens),
        cached=_format_token_count(cached),
        output=_format_token_count(output_tokens),
    )
