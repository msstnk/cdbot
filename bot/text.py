"""Text helpers for Discord limits and Codex SDK payloads."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
# pylint: disable=too-few-public-methods
class ModelDumpable(Protocol):
    """Protocol for SDK models that can produce JSON-compatible dictionaries."""

    @abstractmethod
    def model_dump(self, *, mode: str) -> dict[str, Any]:
        """Return a dictionary representation of the model."""
        raise NotImplementedError


def normalize_user_text(content: str) -> str:
    """Normalize raw Discord message text for prompt handling."""
    return content.strip()


def clip_text(text: str, limit: int) -> str:
    """Clip text to a character limit, using an ellipsis when truncated."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def split_discord_text(text: str, limit: int = 2000) -> list[str]:
    """Split text into Discord-sized chunks while preserving paragraph breaks."""
    if not text:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in _split_long_segments(text.split("\n\n"), limit):
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = paragraph
    if current:
        chunks.append(current)
    return chunks


def extract_assistant_text(turn: object | None) -> str:
    """Extract assistant text from a completed Codex turn object."""
    if turn is None:
        return ""

    chunks: list[str] = []
    for item in getattr(turn, "items", []) or []:
        chunks.extend(extract_text_blocks_from_item(item))
    return "".join(chunks)


def extract_last_assistant_text(turn: object | None) -> str:
    """Extract only the final assistant text item from a completed Codex turn."""
    if turn is None:
        return ""

    last_text = ""
    for item in getattr(turn, "items", []) or []:
        text = "".join(extract_text_blocks_from_item(item))
        if text:
            last_text = text
    return last_text


def extract_text_blocks_from_item(item: object) -> list[str]:
    """Extract assistant text blocks from a Codex item payload."""
    if not isinstance(item, ModelDumpable):
        return []
    payload = item.model_dump(mode="json")

    item_type = payload.get("type")
    if item_type == "agentMessage":
        text = payload.get("text")
        if isinstance(text, str) and text:
            return [text]
        return []

    if item_type != "message" or payload.get("role") != "assistant":
        return []

    blocks: list[str] = []
    for content in payload.get("content") or []:
        if not isinstance(content, dict) or content.get("type") != "output_text":
            continue
        text = content.get("text")
        if isinstance(text, str) and text:
            blocks.append(text)
    return blocks


def _split_long_segments(segments: Iterable[str], limit: int) -> list[str]:
    out: list[str] = []
    for segment in segments:
        if len(segment) <= limit:
            out.append(segment)
            continue

        lines = segment.splitlines(keepends=True)
        if len(lines) > 1:
            current = ""
            for line in lines:
                if len(line) > limit:
                    if current:
                        out.append(current.rstrip("\n"))
                        current = ""
                    out.extend(_split_fixed_width(line, limit))
                    continue
                candidate = current + line
                if len(candidate) <= limit:
                    current = candidate
                    continue
                if current:
                    out.append(current.rstrip("\n"))
                current = line
            if current:
                out.append(current.rstrip("\n"))
            continue

        out.extend(_split_fixed_width(segment, limit))
    return [chunk for chunk in out if chunk]


def _split_fixed_width(text: str, limit: int) -> list[str]:
    return [text[index : index + limit] for index in range(0, len(text), limit)]
