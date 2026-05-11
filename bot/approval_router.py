"""Route Codex approval requests through Discord buttons."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import discord

from .config import DISCORD_MESSAGE_LIMIT
from .localization import Messages, load_default_messages
from .text import clip_text

TURN_APPROVAL_REASON = "turn"
TIMEOUT_REASON = "timed_out"
APPROVAL_INITIAL_CONTENT_LIMIT = 1900
APPROVAL_RESULT_LINE_LIMIT = (
    DISCORD_MESSAGE_LIMIT - APPROVAL_INITIAL_CONTENT_LIMIT - len("\n\n")
)


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """Resolved decision for one approval request."""

    decision: str
    actor_name: str | None
    resolved_at: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """All Discord approval inputs for a single Codex SDK request."""

    channel: discord.abc.Messageable
    requester_id: int
    conversation_key: str
    turn_id: str
    method: str
    params: dict[str, Any] | None


class PendingApproval:
    """Future-backed approval state shared by Discord callbacks."""

    def __init__(self) -> None:
        self._future: asyncio.Future[ApprovalDecision] = (
            asyncio.get_running_loop().create_future()
        )

    def resolve(
        self, decision: str, *, actor_name: str | None, reason: str | None = None
    ) -> bool:
        """Resolve the approval once and report whether this call won."""
        if self._future.done():
            return False
        self._future.set_result(
            ApprovalDecision(
                decision=decision,
                actor_name=actor_name,
                resolved_at=_now_iso(),
                reason=reason,
            )
        )
        return True

    async def wait(self, timeout_sec: float) -> ApprovalDecision:
        """Wait for a decision, denying the request if it times out."""
        try:
            return await asyncio.wait_for(
                asyncio.shield(self._future), timeout=timeout_sec
            )
        except TimeoutError:
            self.resolve("deny", actor_name=None, reason=TIMEOUT_REASON)
            return await self._future


@dataclass(frozen=True, slots=True)
class ApprovalViewState:
    """Per-view state shared across Discord button callbacks."""

    pending: PendingApproval
    requester_id: int
    turn_key: str | None = None


class ApprovalView(discord.ui.View):
    """Discord button view for accepting or denying a Codex request."""

    def __init__(
        self,
        *,
        state: ApprovalViewState,
        turn_approvals: set[str],
        timeout_sec: float,
        messages: Messages,
    ) -> None:
        super().__init__(timeout=timeout_sec)
        self._state = state
        self._turn_approvals = turn_approvals
        self._messages = messages
        self.message: discord.Message | None = None
        self.initial_content = ""
        self._apply_button_labels()

    async def interaction_check(
        self, interaction: discord.Interaction[discord.Client]
    ) -> bool:
        """Allow only the original requester to use approval buttons."""
        if interaction.user.id == self._state.requester_id:
            return True
        if interaction.response.is_done():
            await interaction.followup.send(
                self._messages.text("approval.requester_only"),
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                self._messages.text("approval.requester_only"),
                ephemeral=True,
            )
        return False

    @discord.ui.button(
        label="Approve",
        style=discord.ButtonStyle.success,
        custom_id="approval:approve",
    )
    async def approve(
        self,
        interaction: discord.Interaction[discord.Client],
        _button: discord.ui.Button[discord.ui.View],
    ) -> None:
        """Approve the pending Codex request."""
        decision = ApprovalDecision(
            decision="accept",
            actor_name=interaction.user.display_name,
            resolved_at=_now_iso(),
        )
        self._state.pending.resolve(
            decision.decision,
            actor_name=decision.actor_name,
            reason=decision.reason,
        )
        self.stop()
        await self._finish(interaction, decision)

    @discord.ui.button(
        label="Approve for this turn",
        style=discord.ButtonStyle.primary,
        custom_id="approval:approve_turn",
    )
    async def approve_for_turn(
        self,
        interaction: discord.Interaction[discord.Client],
        _button: discord.ui.Button[discord.ui.View],
    ) -> None:
        """Approve the pending request and future requests in this turn."""
        if self._state.turn_key is not None:
            self._turn_approvals.add(self._state.turn_key)
        decision = ApprovalDecision(
            decision="accept",
            actor_name=interaction.user.display_name,
            resolved_at=_now_iso(),
            reason=TURN_APPROVAL_REASON,
        )
        self._state.pending.resolve(
            decision.decision,
            actor_name=decision.actor_name,
            reason=decision.reason,
        )
        self.stop()
        await self._finish(interaction, decision)

    @discord.ui.button(
        label="Deny", style=discord.ButtonStyle.danger, custom_id="approval:deny"
    )
    async def deny(
        self,
        interaction: discord.Interaction[discord.Client],
        _button: discord.ui.Button[discord.ui.View],
    ) -> None:
        """Deny the pending Codex request."""
        decision = ApprovalDecision(
            decision="deny",
            actor_name=interaction.user.display_name,
            resolved_at=_now_iso(),
        )
        self._state.pending.resolve(
            decision.decision,
            actor_name=decision.actor_name,
            reason=decision.reason,
        )
        self.stop()
        await self._finish(interaction, decision)

    async def on_timeout(self) -> None:
        """Deny and update the Discord message after the view times out."""
        decision = ApprovalDecision(
            decision="deny",
            actor_name=None,
            resolved_at=_now_iso(),
            reason=TIMEOUT_REASON,
        )
        resolved = self._state.pending.resolve(
            decision.decision,
            actor_name=decision.actor_name,
            reason=decision.reason,
        )
        if not resolved:
            return
        await self._edit_message(decision)

    async def finalize(self, decision: ApprovalDecision) -> None:
        """Stop the view and render the resolved decision."""
        self.stop()
        await self._edit_message(decision)

    async def _finish(
        self,
        interaction: discord.Interaction[discord.Client],
        decision: ApprovalDecision,
    ) -> None:
        self._disable_buttons()
        content = self._content_for_decision(decision)
        if interaction.response.is_done():
            if self.message is not None:
                await self.message.edit(content=content, view=self)
            return
        await interaction.response.edit_message(content=content, view=self)

    def _disable_buttons(self) -> None:
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True

    async def _edit_message(self, decision: ApprovalDecision) -> None:
        if self.message is None:
            return
        self._disable_buttons()
        try:
            await self.message.edit(
                content=self._content_for_decision(decision),
                view=self,
            )
        except discord.HTTPException:
            return

    def _content_for_decision(self, decision: ApprovalDecision) -> str:
        return _approval_result_content(self.initial_content, decision, self._messages)

    def _apply_button_labels(self) -> None:
        labels = {
            "approval:approve": self._messages.text("approval.button.approve"),
            "approval:approve_turn": self._messages.text(
                "approval.button.approve_turn"
            ),
            "approval:deny": self._messages.text("approval.button.deny"),
        }
        for child in self.children:
            if isinstance(child, discord.ui.Button) and child.custom_id in labels:
                child.label = labels[child.custom_id]


@dataclass(slots=True)
class ApprovalRouter:
    """Post approval requests to Discord and return Codex SDK decisions."""

    timeout_sec: float
    messages: Messages = field(default_factory=load_default_messages)
    _turn_approvals: set[str] = field(default_factory=set, init=False, repr=False)

    def clear_turn_approval(self, conversation_key: str, turn_id: str) -> None:
        """Forget any in-memory auto-approval granted for a completed turn."""
        turn_key = _turn_approval_key(conversation_key, turn_id)
        if turn_key:
            self._turn_approvals.discard(turn_key)

    async def request_approval(self, request: ApprovalRequest) -> dict[str, str]:
        """Ask Discord for approval and return the SDK decision payload."""
        payload = dict(request.params or {})
        _log_approval_request(request.conversation_key, request.method)
        pending = PendingApproval()
        body = _format_approval_message(request.method, payload, self.messages)
        turn_key = _turn_approval_key(request.conversation_key, request.turn_id)
        if turn_key and turn_key in self._turn_approvals:
            decision = ApprovalDecision(
                decision="accept",
                actor_name=None,
                resolved_at=_now_iso(),
                reason=TURN_APPROVAL_REASON,
            )
            try:
                await request.channel.send(
                    _approval_result_content(body, decision, self.messages)
                )
            except discord.HTTPException:
                pass
            _log_approval_result(request.conversation_key, request.method, decision)
            return {"decision": decision.decision}
        view = ApprovalView(
            state=ApprovalViewState(
                pending=pending,
                requester_id=request.requester_id,
                turn_key=turn_key,
            ),
            turn_approvals=self._turn_approvals,
            timeout_sec=self.timeout_sec,
            messages=self.messages,
        )
        try:
            message = await request.channel.send(body, view=view)
        except discord.HTTPException:
            _log_approval_result(
                request.conversation_key,
                request.method,
                ApprovalDecision(
                    decision="deny",
                    actor_name=None,
                    resolved_at=_now_iso(),
                    reason="discord_post_failed",
                ),
            )
            return {"decision": "deny"}

        view.message = message
        view.initial_content = body
        decision = await pending.wait(self.timeout_sec)
        await view.finalize(decision)
        _log_approval_result(request.conversation_key, request.method, decision)
        return {"decision": decision.decision}


def _format_approval_message(
    method: str,
    params: dict[str, Any],
    messages: Messages | None = None,
) -> str:
    catalog = messages or load_default_messages()
    sections = [catalog.format("approval.required", method=method)]
    command = clip_text(str(params.get("command") or "").strip(), 500)
    if command:
        sections.append(catalog.format("approval.command", command=command))

    cwd = clip_text(str(params.get("cwd") or "").strip(), 300)
    if cwd:
        sections.append(catalog.format("approval.cwd", cwd=cwd))

    reason = clip_text(str(params.get("reason") or "").strip(), 600)
    if reason:
        sections.append(catalog.format("approval.reason", reason=reason))

    command_actions = _format_command_actions(params.get("commandActions"), catalog)
    if command_actions:
        sections.append(
            catalog.format("approval.planned_actions", items=command_actions)
        )

    file_paths = _format_file_paths(params.get("fileChangeFiles"), catalog)
    if file_paths:
        sections.append(file_paths)

    file_change_summary = _format_file_change_changes(
        params.get("fileChangeChanges"),
        limit=_remaining_message_chars(sections),
        messages=catalog,
    )
    if file_change_summary:
        sections.append(file_change_summary)

    details = _format_fallback_details(params, sections, catalog)
    if details:
        sections.append(details)

    return clip_text("\n\n".join(sections), APPROVAL_INITIAL_CONTENT_LIMIT)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _approval_result_content(
    initial_content: str,
    decision: ApprovalDecision,
    messages: Messages | None = None,
) -> str:
    catalog = messages or load_default_messages()
    if decision.decision == "accept" and decision.reason == TURN_APPROVAL_REASON:
        suffix = catalog.text("approval.result.turn")
        reason = ""
    else:
        suffix = (
            catalog.text("approval.result.approved")
            if decision.decision == "accept"
            else catalog.text("approval.result.denied")
        )
        resolved_reason = _resolve_decision_reason(decision.reason, catalog)
        reason = (
            catalog.format("approval.result.reason_suffix", reason=resolved_reason)
            if resolved_reason
            else ""
        )
    actor = (
        catalog.format("approval.result.actor", actor=decision.actor_name)
        if decision.actor_name
        else ""
    )
    result_line = catalog.format(
        "approval.result.line", suffix=suffix, actor=actor, reason=reason
    )
    result_line = clip_text(result_line, APPROVAL_RESULT_LINE_LIMIT)
    available_initial = DISCORD_MESSAGE_LIMIT - len("\n\n") - len(result_line)
    if available_initial <= 0:
        return clip_text(result_line, DISCORD_MESSAGE_LIMIT)
    return f"{clip_text(initial_content, available_initial)}\n\n{result_line}"

def _format_file_paths(value: object, messages: Messages) -> str:
    if not isinstance(value, list):
        return ""
    rendered = "\n".join(
        messages.format("approval.file_path.item", path=clip_text(path, 180))
        for path in (
            str(entry).strip()
            for entry in value[:10]
        )
        if path
    )
    if not rendered:
        return ""
    return messages.format("approval.files", items=rendered)


def _format_fallback_details(
    params: dict[str, Any],
    sections: list[str],
    messages: Messages,
) -> str:
    if len(sections) != 1:
        return ""
    details_limit = _remaining_message_chars(sections)
    template = messages.text("approval.details")
    prefix, _, suffix = template.partition("{serialized}")
    serialized = clip_text(
        json.dumps(params, ensure_ascii=False, indent=2),
        max(0, details_limit - len(prefix) - len(suffix)),
    )
    if not serialized:
        return ""
    return messages.format("approval.details", serialized=serialized)


def _log_approval_request(conversation_key: str, method: str) -> None:
    print(
        json.dumps(
            {
                "type": "approval_request",
                "conversation_key": conversation_key,
                "method": method,
                "requested_at": _now_iso(),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


def _log_approval_result(
    conversation_key: str,
    method: str,
    decision: ApprovalDecision,
) -> None:
    print(
        json.dumps(
            {
                "type": "approval_result",
                "conversation_key": conversation_key,
                "method": method,
                "decision": decision.decision,
                "actor": decision.actor_name,
                "reason": decision.reason,
                "resolved_at": decision.resolved_at,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


def _format_command_actions(value: object, messages: Messages) -> str:
    if not isinstance(value, list):
        return ""
    lines: list[str] = []
    for entry in value[:5]:
        if not isinstance(entry, dict):
            continue
        command = clip_text(str(entry.get("command") or "").strip(), 220)
        action_type = str(entry.get("type") or "unknown").strip() or "unknown"
        path = clip_text(str(entry.get("path") or "").strip(), 140)
        query = clip_text(str(entry.get("query") or "").strip(), 140)
        details = [messages.format("approval.command_action.type", value=action_type)]
        if path:
            details.append(messages.format("approval.command_action.path", value=path))
        if query:
            details.append(
                messages.format("approval.command_action.query", value=query)
            )
        if command:
            details.append(
                messages.format("approval.command_action.command", value=command)
            )
        lines.append(f"- {', '.join(details)}")
    return "\n".join(lines)


def _remaining_message_chars(
    sections: list[str], limit: int = APPROVAL_INITIAL_CONTENT_LIMIT
) -> int:
    current = "\n\n".join(section for section in sections if section)
    if not current:
        return limit
    return max(0, limit - len(current) - len("\n\n"))


def _format_file_change_changes(
    value: object,
    *,
    limit: int,
    messages: Messages,
) -> str:
    if not isinstance(value, list):
        return ""
    title = messages.format("approval.proposed_changes", items="")
    remaining = limit - len(title)
    if remaining <= 0:
        return ""
    rendered: list[str] = []
    for entry in value[:5]:
        if not isinstance(entry, dict):
            continue
        separator = "\n" if rendered else ""
        entry_limit = remaining - len(separator)
        if entry_limit <= 0:
            break
        path = clip_text(str(entry.get("path") or "").strip(), 180)
        if not path:
            continue
        kind = clip_text(str(entry.get("kind") or "-").strip() or "-", 40)
        snippet = str(entry.get("snippet") or "").strip()
        entry_text = messages.format("approval.file_change.item", path=path, kind=kind)
        if snippet:
            fixed_overhead = len(entry_text) + len("\n```diff\n") + len("\n```")
            snippet_limit = entry_limit - fixed_overhead
            if snippet_limit > 0:
                entry_text = (
                    f"{entry_text}\n```diff\n"
                    f"{clip_text(str(entry.get('snippet') or '').strip(), snippet_limit)}\n```"
                )
        entry_text = clip_text(entry_text, entry_limit)
        rendered.append(entry_text)
        remaining -= len(separator) + len(entry_text)
    if not rendered:
        return ""
    return messages.format("approval.proposed_changes", items="\n".join(rendered))


def _resolve_decision_reason(reason: str | None, messages: Messages) -> str:
    if reason == TIMEOUT_REASON:
        return messages.text("approval.reason.timed_out")
    if reason == TURN_APPROVAL_REASON:
        return ""
    return reason or ""


def _turn_approval_key(conversation_key: str, turn_id: str) -> str:
    if not conversation_key or not turn_id:
        return ""
    return f"{conversation_key}:{turn_id}"
