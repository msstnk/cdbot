"""Tests for Discord text splitting and Codex text extraction helpers."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

from __future__ import annotations

from types import SimpleNamespace

from bot.text import (
    clip_text,
    extract_assistant_text,
    extract_text_blocks_from_item,
    split_discord_text,
)


class DumpableItem:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return self.payload


def test_split_discord_text_returns_no_chunks_for_empty_text() -> None:
    assert not split_discord_text("")


def test_split_discord_text_preserves_paragraphs_when_chunking() -> None:
    text = "alpha\n\nbeta\n\ngamma"

    chunks = split_discord_text(text, limit=11)

    assert chunks == ["alpha\n\nbeta", "gamma"]


def test_split_discord_text_splits_long_single_line_fixed_width() -> None:
    chunks = split_discord_text("abcdefghij", limit=4)

    assert chunks == ["abcd", "efgh", "ij"]


def test_split_discord_text_splits_long_multiline_segment_on_line_boundaries() -> None:
    text = "alpha\nbeta\ngamma"

    chunks = split_discord_text(text, limit=10)

    assert chunks == ["alpha", "beta\ngamma"]


def test_clip_text_uses_ellipsis_when_truncated() -> None:
    assert clip_text("abcdef", 5) == "ab..."


def test_extract_text_blocks_from_agent_message() -> None:
    item = {"type": "agentMessage", "text": "hello"}

    blocks = extract_text_blocks_from_item(item)

    assert blocks == ["hello"]


def test_extract_text_blocks_from_assistant_output_text_content() -> None:
    item = {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": "hello"},
            {"type": "input_text", "text": "ignored"},
            {"type": "output_text", "text": " world"},
        ],
    }

    blocks = extract_text_blocks_from_item(item)

    assert blocks == ["hello", " world"]


def test_extract_text_blocks_ignores_non_assistant_messages() -> None:
    item = {
        "type": "message",
        "role": "user",
        "content": [{"type": "output_text", "text": "ignored"}],
    }

    blocks = extract_text_blocks_from_item(item)

    assert not blocks


def test_extract_assistant_text_supports_model_dumpable_items() -> None:
    turn = SimpleNamespace(
        items=[
            DumpableItem({"type": "agentMessage", "text": "hello"}),
            DumpableItem(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": " world"}],
                }
            ),
        ]
    )

    text = extract_assistant_text(turn)

    assert text == "hello world"
