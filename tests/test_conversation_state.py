from __future__ import annotations

import pytest

from bot.conversation_state import ActiveTurn, ConversationState


@pytest.mark.asyncio
async def test_active_turn_steer_messages_are_merged() -> None:
    active_turn = ActiveTurn(conversation_key="dm:1")
    active_turn.queue_steer("first")
    active_turn.queue_steer("second")

    assert active_turn.drain_steer() == "first\n\nsecond"
    assert active_turn.drain_steer() == ""


@pytest.mark.asyncio
async def test_conversation_state_tracks_single_active_turn() -> None:
    state = ConversationState()
    first = ActiveTurn(conversation_key="dm:1")
    second = ActiveTurn(conversation_key="dm:1")

    assert await state.start(first) is True
    assert await state.start(second) is False
    assert await state.get("dm:1") is first

    await state.finish(first)
    assert await state.get("dm:1") is None
