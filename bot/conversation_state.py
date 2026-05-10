"""Track in-flight Codex turns for each Discord conversation."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field


@dataclass(slots=True)
class TurnIds:
    """Codex identifiers assigned while a turn is running."""

    thread_id: str = ""
    turn_id: str = ""


@dataclass(slots=True)
class ActiveTurn:
    """Mutable state shared between a running turn and incoming Discord messages."""

    conversation_key: str
    steer_messages: list[str] = field(default_factory=list)
    task: asyncio.Task[None] | None = None
    ids: TurnIds = field(default_factory=TurnIds)
    discard_output: bool = False
    _interrupt_requested: bool = False
    _steer_lock: threading.Lock = field(default_factory=threading.Lock)

    def queue_steer(self, text: str) -> None:
        """Queue extra user text to steer the active turn."""
        with self._steer_lock:
            self.steer_messages.append(text)

    def drain_steer(self) -> str:
        """Return queued steer text and clear the queue."""
        with self._steer_lock:
            if not self.steer_messages:
                return ""
            merged = "\n\n".join(self.steer_messages)
            self.steer_messages.clear()
            return merged

    def request_interrupt(self) -> None:
        """Mark the turn so the runner sends one interrupt request."""
        self._interrupt_requested = True

    def take_interrupt_request(self) -> bool:
        """Return and clear the pending interrupt request."""
        requested = self._interrupt_requested
        self._interrupt_requested = False
        return requested


class ConversationState:
    """Async-safe registry of active turns by conversation key."""

    def __init__(self) -> None:
        self._active_turns: dict[str, ActiveTurn] = {}
        self._lock = asyncio.Lock()

    async def get(self, conversation_key: str) -> ActiveTurn | None:
        """Return the active turn for a conversation, if one exists."""
        async with self._lock:
            return self._active_turns.get(conversation_key)

    async def start(self, active_turn: ActiveTurn) -> bool:
        """Register a turn unless the conversation already has one running."""
        async with self._lock:
            if active_turn.conversation_key in self._active_turns:
                return False
            self._active_turns[active_turn.conversation_key] = active_turn
            return True

    async def finish(self, active_turn: ActiveTurn) -> None:
        """Remove a turn only if it is still the registered active turn."""
        async with self._lock:
            existing = self._active_turns.get(active_turn.conversation_key)
            if existing is active_turn:
                self._active_turns.pop(active_turn.conversation_key, None)

    async def clear(self, conversation_key: str) -> ActiveTurn | None:
        """Remove and return the active turn for a conversation."""
        async with self._lock:
            return self._active_turns.pop(conversation_key, None)
