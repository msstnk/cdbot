from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from bot.approval_router import ApprovalRequest, ApprovalRouter
from bot.codex_runner import (
    CodexRunner,
    RunnerDependencies,
    RunnerEnvironment,
    SessionContext,
    TurnContext,
    TurnRequest,
    _build_app_server_env,
)
from bot.conversation_state import ActiveTurn


@dataclass
class FakeNotification:
    method: str
    payload: Any


class FakeStatus:
    def __init__(self, value: str) -> None:
        self.value = value


class FakeClient:
    def __init__(self, *, resume_error: bool = False) -> None:
        self.resume_error = resume_error
        self.turn_steer_calls: list[str] = []
        self.thread_resume_calls = 0
        self.thread_start_calls = 0
        self.turn_start_inputs: list[str] = []
        self.notifications = [
            FakeNotification(
                method="item/completed",
                payload=SimpleNamespace(
                    turn_id="turn-1",
                    item=SimpleNamespace(
                        model_dump=lambda mode="json": {
                            "type": "agentMessage",
                            "text": "hello from item",
                        }
                    ),
                ),
            ),
            FakeNotification(
                method="turn/completed",
                payload=SimpleNamespace(
                    turn=SimpleNamespace(
                        id="turn-1",
                        status=FakeStatus("completed"),
                        items=[],
                    )
                ),
            )
        ]

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def initialize(self) -> object:
        return object()

    def thread_start(self, params: dict[str, Any] | None = None) -> object:
        self.thread_start_calls += 1
        return SimpleNamespace(thread=SimpleNamespace(id="thread-new"))

    def thread_resume(self, thread_id: str, params: dict[str, Any] | None = None) -> object:
        self.thread_resume_calls += 1
        if self.resume_error:
            raise RuntimeError("missing rollout")
        return SimpleNamespace(thread=SimpleNamespace(id=thread_id))

    def turn_start(
        self,
        thread_id: str,
        input_items: str,
        params: dict[str, Any] | None = None,
    ) -> object:
        self.turn_start_inputs.append(input_items)
        return SimpleNamespace(turn=SimpleNamespace(id="turn-1"))

    def turn_steer(self, thread_id: str, expected_turn_id: str, input_items: str) -> object:
        self.turn_steer_calls.append(input_items)
        return object()

    def turn_interrupt(self, thread_id: str, turn_id: str) -> object:
        return object()

    def next_notification(self) -> object:
        return self.notifications.pop(0)


class ApprovalCaptureRouter:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def clear_turn_approval(self, conversation_key: str, turn_id: str) -> None:
        _ = (conversation_key, turn_id)

    async def request_approval(
        self, request: ApprovalRequest
    ) -> dict[str, str]:
        self.requests.append(
            {
                "channel": request.channel,
                "requester_id": request.requester_id,
                "conversation_key": request.conversation_key,
                "turn_id": request.turn_id,
                "method": request.method,
                "params": request.params,
            }
        )
        return {"decision": "accept"}


class FakeApprovalClient(FakeClient):
    def __init__(
        self,
        approval_handler: Any,
        *,
        diff: str = "@@ -1 +1 @@\n-print('old')\n+print('new')",
    ) -> None:
        super().__init__(resume_error=False)
        self._approval_handler = approval_handler
        self._approval_requested = False
        self.notifications = [
            FakeNotification(
                method="item/started",
                payload=SimpleNamespace(
                    turn_id="turn-1",
                    item=SimpleNamespace(
                        model_dump=lambda mode="json": {
                            "id": "fc-1",
                            "type": "fileChange",
                            "changes": [
                                {
                                    "path": "bot/approval_router.py",
                                    "diff": diff,
                                    "kind": {"type": "update"},
                                }
                            ],
                        }
                    ),
                ),
            ),
            FakeNotification(
                method="turn/completed",
                payload=SimpleNamespace(
                    turn=SimpleNamespace(
                        id="turn-1",
                        status=FakeStatus("completed"),
                        items=[],
                    )
                ),
            ),
        ]
        self.approval_result: dict[str, Any] | None = None

    def next_notification(self) -> object:
        if not self._approval_requested and self.notifications:
            next_method = self.notifications[0].method
            if next_method == "turn/completed":
                self.approval_result = self._approval_handler(
                    "item/fileChange/requestApproval",
                    {"itemId": "fc-1"},
                )
                self._approval_requested = True
        notification = self.notifications.pop(0)
        return notification


class DummyChannel:
    async def send(self, *args: Any, **kwargs: Any) -> object:
        return object()


def _turn_request(
    *,
    prompt: str = "hello",
    model: str = "gpt-5.4",
    cwd: str = "/tmp/project",
    session_id: str = "",
    session_model: str = "",
    session_cwd: str = "",
) -> TurnRequest:
    return TurnRequest(
        context=TurnContext(
            conversation_key="dm:1",
            requester_id=1,
            channel=DummyChannel(),
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


def _runner(
    *,
    loop: asyncio.AbstractEventLoop,
    approval_router: ApprovalRouter | ApprovalCaptureRouter | None = None,
    client_factory: Any = None,
) -> CodexRunner:
    return CodexRunner(
        environment=RunnerEnvironment(
            codex_bin="/tmp/codex",
            codex_home="/tmp/.codex",
            workspace_cwd="/tmp",
        ),
        dependencies=RunnerDependencies(
            approval_router=cast(
                ApprovalRouter,
                approval_router or ApprovalRouter(timeout_sec=0.1),
            ),
            loop=loop,
            client_factory=client_factory,
        ),
    )


def test_runner_builds_config_with_request_cwd() -> None:
    runner = CodexRunner(
        environment=RunnerEnvironment(
            codex_bin="/tmp/codex",
            codex_home="/tmp/.codex",
            workspace_cwd="/tmp/default",
        ),
        dependencies=RunnerDependencies(
            approval_router=ApprovalRouter(timeout_sec=0.1),
            loop=asyncio.new_event_loop(),
        ),
    )

    config = runner._build_config("/tmp/project")

    assert config.cwd == "/tmp/project"
    assert config.env == {"CODEX_HOME": "/tmp/.codex"}


def test_build_app_server_env_removes_cdbot_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CDBOT_DISCORD_BOT_TOKEN", "secret")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("OPENAI_API_KEY", "visible")

    env = _build_app_server_env({"CODEX_HOME": "/tmp/.codex"})

    assert env["PATH"] == "/usr/bin"
    assert env["OPENAI_API_KEY"] == "visible"
    assert env["CODEX_HOME"] == "/tmp/.codex"
    assert "CDBOT_DISCORD_BOT_TOKEN" not in env


@pytest.mark.asyncio
async def test_runner_falls_back_to_thread_start_when_resume_fails() -> None:
    loop = asyncio.get_running_loop()
    fake_client = FakeClient(resume_error=True)
    runner = _runner(
        loop=loop,
        client_factory=lambda _config, _approval_handler: fake_client,
    )
    active_turn = ActiveTurn(conversation_key="dm:1")
    request = _turn_request(
        session_id="thread-old",
        session_model="gpt-5.4",
        session_cwd="/tmp/project",
    )

    result = await runner.run_turn(request, active_turn, _ignore_delta)

    assert result.thread_id == "thread-new"
    assert result.resumed is False
    assert fake_client.thread_resume_calls == 1
    assert fake_client.thread_start_calls == 1
    assert fake_client.turn_start_inputs == [
        "Current working directory: /tmp/project\n\nhello"
    ]


@pytest.mark.asyncio
async def test_runner_merges_multiple_steer_messages() -> None:
    loop = asyncio.get_running_loop()
    fake_client = FakeClient(resume_error=False)
    runner = _runner(
        loop=loop,
        client_factory=lambda _config, _approval_handler: fake_client,
    )
    active_turn = ActiveTurn(conversation_key="dm:1")
    active_turn.queue_steer("first")
    active_turn.queue_steer("second")
    request = _turn_request()

    await runner.run_turn(request, active_turn, _ignore_delta)

    assert fake_client.turn_steer_calls == ["first\n\nsecond"]
    assert fake_client.turn_start_inputs == [
        "Current working directory: /tmp/project\n\nhello"
    ]


@pytest.mark.asyncio
async def test_runner_emits_blocks_from_item_completed() -> None:
    loop = asyncio.get_running_loop()
    fake_client = FakeClient(resume_error=False)
    runner = _runner(
        loop=loop,
        client_factory=lambda _config, _approval_handler: fake_client,
    )
    active_turn = ActiveTurn(conversation_key="dm:1")
    request = _turn_request()
    blocks: list[str] = []

    result = await runner.run_turn(request, active_turn, lambda block: _record_block(blocks, block))

    assert blocks == ["hello from item"]
    assert result.assistant_text == "hello from item"


@pytest.mark.asyncio
async def test_runner_does_not_redeclare_cwd_on_resumed_session() -> None:
    loop = asyncio.get_running_loop()
    fake_client = FakeClient(resume_error=False)
    runner = _runner(
        loop=loop,
        client_factory=lambda _config, _approval_handler: fake_client,
    )
    active_turn = ActiveTurn(conversation_key="dm:1")
    request = _turn_request(
        session_id="thread-old",
        session_model="gpt-5.4",
        session_cwd="/tmp/project",
    )

    await runner.run_turn(request, active_turn, _ignore_delta)

    assert fake_client.turn_start_inputs == ["hello"]


@pytest.mark.asyncio
async def test_runner_redeclares_cwd_when_resumed_session_changed_directory() -> None:
    loop = asyncio.get_running_loop()
    fake_client = FakeClient(resume_error=False)
    runner = _runner(
        loop=loop,
        client_factory=lambda _config, _approval_handler: fake_client,
    )
    active_turn = ActiveTurn(conversation_key="dm:1")
    request = _turn_request(
        cwd="/tmp/updated",
        session_id="thread-old",
        session_model="gpt-5.4",
        session_cwd="/tmp/original",
    )

    await runner.run_turn(request, active_turn, _ignore_delta)

    assert fake_client.turn_start_inputs == [
        "Current working directory: /tmp/updated\n\nhello"
    ]


@pytest.mark.asyncio
async def test_runner_starts_new_thread_when_resumed_session_changed_model() -> None:
    loop = asyncio.get_running_loop()
    fake_client = FakeClient(resume_error=False)
    runner = _runner(
        loop=loop,
        client_factory=lambda _config, _approval_handler: fake_client,
    )
    active_turn = ActiveTurn(conversation_key="dm:1")
    request = _turn_request(
        model="gpt-5.5",
        session_id="thread-old",
        session_model="gpt-5.4",
        session_cwd="/tmp/project",
    )

    result = await runner.run_turn(request, active_turn, _ignore_delta)

    assert result.thread_id == "thread-new"
    assert result.resumed is False
    assert fake_client.thread_resume_calls == 0
    assert fake_client.thread_start_calls == 1
    assert fake_client.turn_start_inputs == ["Current working directory: /tmp/project\n\nhello"]


@pytest.mark.asyncio
async def test_runner_attaches_file_change_preview_to_approval_request() -> None:
    loop = asyncio.get_running_loop()
    approval_router = ApprovalCaptureRouter()
    runner = _runner(
        loop=loop,
        approval_router=approval_router,
        client_factory=lambda _config, approval_handler: FakeApprovalClient(approval_handler),
    )
    active_turn = ActiveTurn(conversation_key="dm:1")
    request = _turn_request()

    await runner.run_turn(request, active_turn, _ignore_delta)

    assert approval_router.requests == [
        {
            "channel": request.context.channel,
            "requester_id": 1,
            "conversation_key": "dm:1",
            "turn_id": "turn-1",
            "method": "item/fileChange/requestApproval",
            "params": {
                "itemId": "fc-1",
                "fileChangeFiles": ["bot/approval_router.py"],
                "fileChangeChanges": [
                    {
                        "path": "bot/approval_router.py",
                        "kind": "update",
                        "snippet": "@@ -1 +1 @@\n-print('old')\n+print('new')",
                    }
                ],
            },
        }
    ]


@pytest.mark.asyncio
async def test_runner_preserves_full_file_change_diff_for_approval_preview() -> None:
    loop = asyncio.get_running_loop()
    approval_router = ApprovalCaptureRouter()
    diff = "@@ -1 +1,80 @@\n" + ("+print('new')\n" * 80)
    runner = _runner(
        loop=loop,
        approval_router=approval_router,
        client_factory=lambda _config, approval_handler: FakeApprovalClient(
            approval_handler,
            diff=diff,
        ),
    )
    active_turn = ActiveTurn(conversation_key="dm:1")
    request = _turn_request()

    await runner.run_turn(request, active_turn, _ignore_delta)

    preview = approval_router.requests[0]["params"]["fileChangeChanges"][0]
    assert preview["snippet"] == diff.strip()


async def _ignore_delta(_: str) -> None:
    return None


async def _record_block(blocks: list[str], block: str) -> None:
    blocks.append(block)
