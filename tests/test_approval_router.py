from __future__ import annotations

from typing import Any, cast

import pytest

from bot.approval_router import (
    APPROVAL_INITIAL_CONTENT_LIMIT,
    TIMEOUT_REASON,
    TURN_APPROVAL_REASON,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalRouter,
    ApprovalView,
    ApprovalViewState,
    PendingApproval,
    _approval_result_content,
    _format_approval_message,
)
from bot.config import DISCORD_MESSAGE_LIMIT
from bot.localization import Messages


@pytest.mark.asyncio
async def test_pending_approval_accept() -> None:
    pending = PendingApproval()
    pending.resolve("accept", actor_name="alice")

    decision = await pending.wait(0.1)

    assert decision.decision == "accept"
    assert decision.actor_name == "alice"


@pytest.mark.asyncio
async def test_pending_approval_deny() -> None:
    pending = PendingApproval()
    pending.resolve("deny", actor_name="bob")

    decision = await pending.wait(0.1)

    assert decision.decision == "deny"
    assert decision.actor_name == "bob"


@pytest.mark.asyncio
async def test_pending_approval_timeout_defaults_to_deny() -> None:
    pending = PendingApproval()

    decision = await pending.wait(0.01)

    assert decision.decision == "deny"
    assert decision.reason == TIMEOUT_REASON


class DummyMessage:
    def __init__(self) -> None:
        self.content = "approval"
        self.edits: list[str] = []

    async def edit(self, *, content: str | None = None, view: object | None = None) -> None:
        if content is not None:
            self.content = content
            self.edits.append(content)
        _ = view


class DummyChannel:
    def __init__(self) -> None:
        self.messages: list[tuple[str, object | None]] = []

    async def send(self, content: str, view: object | None = None) -> DummyMessage:
        self.messages.append((content, view))
        message = DummyMessage()
        message.content = content
        return message


class DummyResponse:
    def __init__(self) -> None:
        self.edited_content = ""

    def is_done(self) -> bool:
        return False

    async def edit_message(self, *, content: str | None = None, view: object | None = None) -> None:
        if content is not None:
            self.edited_content = content
        _ = view

    async def send_message(self, content: str, ephemeral: bool = False) -> None:
        _ = (content, ephemeral)


class DummyFollowup:
    async def send(self, content: str, ephemeral: bool = False) -> None:
        _ = (content, ephemeral)


class DummyUser:
    def __init__(self, user_id: int, display_name: str) -> None:
        self.id = user_id
        self.display_name = display_name


class DummyInteraction:
    def __init__(self, user_id: int, display_name: str) -> None:
        self.user = DummyUser(user_id, display_name)
        self.response = DummyResponse()
        self.followup = DummyFollowup()


@pytest.mark.asyncio
async def test_timeout_does_not_override_existing_decision() -> None:
    pending = PendingApproval()
    view = ApprovalView(
        state=ApprovalViewState(pending=pending, requester_id=1),
        turn_approvals=set(),
        timeout_sec=60.0,
        messages=_messages(),
    )
    message = DummyMessage()
    view.message = cast(Any, message)
    pending.resolve("accept", actor_name="alice")

    await view.on_timeout()

    assert message.edits == []


@pytest.mark.asyncio
async def test_approve_for_turn_enables_turn_mode() -> None:
    pending = PendingApproval()
    turn_approvals: set[str] = set()
    view = ApprovalView(
        state=ApprovalViewState(
            pending=pending,
            requester_id=1,
            turn_key="dm:1:turn-1",
        ),
        turn_approvals=turn_approvals,
        timeout_sec=60.0,
        messages=_messages(),
    )
    view.initial_content = "approval"
    button = next(
        child
        for child in view.children
        if getattr(child, "label", "") == _messages().text("approval.button.approve_turn")
    )
    interaction = DummyInteraction(1, "alice")

    await button.callback(cast(Any, interaction))
    decision = await pending.wait(0.1)

    assert turn_approvals == {"dm:1:turn-1"}
    assert decision.decision == "accept"
    assert decision.reason == TURN_APPROVAL_REASON
    assert (
        interaction.response.edited_content
        == f"approval\n\n{_result_line('approval.result.turn', actor_name='alice')}"
    )


@pytest.mark.asyncio
async def test_request_approval_auto_approves_when_turn_mode_enabled() -> None:
    channel = DummyChannel()
    router = ApprovalRouter(timeout_sec=60.0, messages=_messages())
    router._turn_approvals.add("dm:1:turn-1")

    result = await router.request_approval(
        ApprovalRequest(
            channel=cast(Any, channel),
            requester_id=1,
            conversation_key="dm:1",
            turn_id="turn-1",
            method="shell.exec",
            params={"command": "pwd"},
        )
    )

    assert result == {"decision": "accept"}
    assert channel.messages == [
        (
            (
                f"{_messages().format('approval.required', method='shell.exec')}\n\n"
                f"{_messages().format('approval.command', command='pwd')}\n\n"
                f"{_result_line('approval.result.turn')}"
            ),
            None,
        )
    ]


def test_format_approval_message_includes_command_actions() -> None:
    message = _format_approval_message(
        "item/commandExecution/requestApproval",
        {
            "command": "rg approval bot",
            "cwd": "/workspace",
            "commandActions": [
                {"type": "search", "path": "bot", "query": "approval", "command": "rg approval bot"}
            ],
        },
        _messages(),
    )

    assert _messages().text("approval.planned_actions").split(":\n", maxsplit=1)[0] in message
    assert "type=search" in message
    assert "path=`bot`" in message
    assert "query=`approval`" in message


def test_format_approval_message_includes_file_change_snippets() -> None:
    message = _format_approval_message(
        "item/fileChange/requestApproval",
        {
            "fileChangeFiles": ["bot/approval_router.py"],
            "fileChangeChanges": [
                {
                    "path": "bot/approval_router.py",
                    "kind": "update",
                    "snippet": "@@ -1,2 +1,3 @@\n+print('hi')",
                }
            ],
        },
        _messages(),
    )

    assert _messages().text("approval.files").split(":\n", maxsplit=1)[0] in message
    assert _messages().text("approval.proposed_changes").split(":\n", maxsplit=1)[0] in message
    assert "```diff" in message
    assert "print('hi')" in message


def test_format_approval_message_keeps_full_file_change_snippet_when_it_fits() -> None:
    snippet = "@@ -1,2 +1,40 @@\n" + ("+print('hi')\n" * 40)
    message = _format_approval_message(
        "item/fileChange/requestApproval",
        {
            "fileChangeFiles": ["bot/approval_router.py"],
            "fileChangeChanges": [
                {
                    "path": "bot/approval_router.py",
                    "kind": "update",
                    "snippet": snippet,
                }
            ],
        },
        _messages(),
    )

    assert snippet in message


def test_format_approval_message_reserves_result_space_for_file_changes() -> None:
    message = _format_approval_message(
        "item/fileChange/requestApproval",
        {
            "fileChangeFiles": ["bot/approval_router.py"],
            "fileChangeChanges": [
                {
                    "path": "bot/approval_router.py",
                    "kind": "update",
                    "snippet": "@@ -1,2 +1,200 @@\n" + ("+print('hi')\n" * 400),
                }
            ],
        },
        _messages(),
    )

    assert len(message) == APPROVAL_INITIAL_CONTENT_LIMIT
    assert _messages().text("approval.proposed_changes").split(":\n", maxsplit=1)[0] in message
    assert "```diff" in message


def test_approval_result_content_clips_initial_message_to_discord_limit() -> None:
    content = _approval_result_content(
        "x" * DISCORD_MESSAGE_LIMIT,
        ApprovalDecision(
            decision="accept",
            actor_name="alice",
            resolved_at="2026-05-10T00:00:00+00:00",
        ),
        _messages(),
    )

    assert len(content) == DISCORD_MESSAGE_LIMIT
    assert content.endswith(_result_line("approval.result.approved", actor_name="alice"))


def test_approval_result_content_preserves_file_change_code_fence() -> None:
    message = _format_approval_message(
        "item/fileChange/requestApproval",
        {
            "fileChangeFiles": ["bot/approval_router.py"],
            "fileChangeChanges": [
                {
                    "path": "bot/approval_router.py",
                    "kind": "update",
                    "snippet": "@@ -1,2 +1,200 @@\n" + ("+print('hi')\n" * 400),
                }
            ],
        },
        _messages(),
    )

    content = _approval_result_content(
        message,
        ApprovalDecision(
            decision="accept",
            actor_name="alice",
            resolved_at="2026-05-10T00:00:00+00:00",
        ),
        _messages(),
    )

    assert len(content) <= DISCORD_MESSAGE_LIMIT
    assert content.count("```") % 2 == 0
    assert content.endswith(_result_line("approval.result.approved", actor_name="alice"))


def _messages() -> Messages:
    return Messages.load("ja_JP")


def _result_line(
    suffix_key: str,
    *,
    actor_name: str | None = None,
    reason_key: str | None = None,
) -> str:
    messages = _messages()
    actor = (
        messages.format("approval.result.actor", actor=actor_name)
        if actor_name is not None
        else ""
    )
    reason = (
        messages.format("approval.result.reason_suffix", reason=messages.text(reason_key))
        if reason_key is not None
        else ""
    )
    return messages.format(
        "approval.result.line",
        suffix=messages.text(suffix_key),
        actor=actor,
        reason=reason,
    )
