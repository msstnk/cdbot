"""OpenAI-backed transcription for Discord voice message attachments."""
# pylint: disable=import-outside-toplevel,too-few-public-methods

from __future__ import annotations

from pathlib import Path
from typing import Any

from .debug_logging import get_logger


class OpenAIVoiceTranscriber:
    """Transcribe Discord voice attachments with the OpenAI audio API."""

    def __init__(self, *, api_key: str, model: str) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._logger = get_logger()
        self._model = model

    async def transcribe_attachment(self, attachment: Any) -> str:
        """Return a text transcript for one Discord voice attachment."""
        filename = _normalized_filename(
            getattr(attachment, "filename", ""),
            getattr(attachment, "content_type", None),
        )
        media_type = getattr(attachment, "content_type", None) or "audio/ogg"
        size = getattr(attachment, "size", 0)
        self._logger.debug(
            "discord.voice.transcription.start attachment_id=%s filename=%s size=%s model=%s",
            getattr(attachment, "id", "-"),
            filename,
            size,
            self._model,
        )
        try:
            audio_bytes = await attachment.read()
            transcription = await self._client.audio.transcriptions.create(
                file=(filename, audio_bytes, media_type),
                model=self._model,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise RuntimeError("OpenAI voice transcription request failed") from exc
        text = getattr(transcription, "text", "")
        if not isinstance(text, str):
            raise RuntimeError("OpenAI transcription response did not include text")
        normalized = text.strip()
        self._logger.debug(
            "discord.voice.transcription.ok attachment_id=%s transcript_chars=%s",
            getattr(attachment, "id", "-"),
            len(normalized),
        )
        return normalized


def _normalized_filename(filename: str, content_type: object) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix:
        return filename

    if isinstance(content_type, str):
        if content_type == "audio/ogg":
            return "voice.ogg"
        if content_type == "audio/webm":
            return "voice.webm"
        if content_type == "audio/mpeg":
            return "voice.mp3"
        if content_type == "audio/wav":
            return "voice.wav"

    return "voice.ogg"
