from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import Any, cast

import pytest
from openai_codex import AppServerConfig
from openai_codex.types import ThreadTokenUsageUpdatedNotification

from bot.approval_router import ApprovalRequest, ApprovalRouter
from bot.codex_runner import (
    AssistantOutputBlock,
    CodexRunner,
    RunnerDependencies,
    RunnerEnvironment,
    SanitizedAppServerClient,
    SessionContext,
    TurnContext,
    TurnRequest,
    _build_app_server_env,
)
from bot.conversation_state import ActiveTurn
from bot.debug_logging import configure_debug_logger


@dataclass
class FakeNotification:
    method: str
    payload: Any


class FakeStatus:
    def __init__(self, value: str) -> None:
        self.value = value


def _token_usage_payload(
    *,
    cached_input_tokens: int,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
) -> ThreadTokenUsageUpdatedNotification:
    usage = {
        "cachedInputTokens": cached_input_tokens,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "reasoningOutputTokens": 0,
        "totalTokens": total_tokens,
    }
    return ThreadTokenUsageUpdatedNotification.model_validate(
        {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "tokenUsage": {
                "last": usage,
                "total": usage,
            },
        }
    )


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
                method="thread/tokenUsage/updated",
                payload=_token_usage_payload(
                    cached_input_tokens=10,
                    input_tokens=20,
                    output_tokens=30,
                    total_tokens=50,
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

    def next_turn_notification(self, turn_id: str) -> object:
        _ = turn_id
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

    def next_turn_notification(self, turn_id: str) -> object:
        _ = turn_id
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

# pylint: disable=too-many-arguments
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


# pylint: enable=too-many-arguments


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


def test_runner_builds_config_with_rust_log_from_debug_level() -> None:
    runner = CodexRunner(
        environment=RunnerEnvironment(
            codex_bin="/tmp/codex",
            codex_home="/tmp/.codex",
            workspace_cwd="/tmp/default",
            debug_level_name="DEBUG",
        ),
        dependencies=RunnerDependencies(
            approval_router=ApprovalRouter(timeout_sec=0.1),
            loop=asyncio.new_event_loop(),
        ),
    )

    config = runner._build_config("/tmp/project")

    assert config.env == {"CODEX_HOME": "/tmp/.codex", "RUST_LOG": "debug"}


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


def test_sanitized_client_start_starts_reader_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePopen:
        def __init__(self) -> None:
            self.stdin = object()
            self.stdout = object()
            self.stderr = object()
            self.pid = 123

        def __enter__(self) -> FakePopen:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            _ = (exc_type, exc, tb)

    def fake_popen(*args: Any, **kwargs: Any) -> FakePopen:
        _ = (args, kwargs)
        return FakePopen()

    monkeypatch.setattr(
        "bot.codex_runner._resolve_codex_bin",
        lambda _config: Path("/tmp/codex"),
    )
    monkeypatch.setattr("bot.codex_runner.subprocess.Popen", fake_popen)
    client = SanitizedAppServerClient(
        config=AppServerConfig(
            codex_bin="/tmp/codex",
            cwd="/tmp/project",
            env={"CODEX_HOME": "/tmp/.codex"},
        )
    )
    started: list[str] = []
    client._start_stderr_drain_thread = lambda: started.append("stderr")  # type: ignore[method-assign]
    client._start_reader_thread = lambda: started.append("reader")  # type: ignore[method-assign]

    client.start()

    assert started == ["stderr", "reader"]


def test_sanitized_client_stderr_logs_at_trace(tmp_path: Path) -> None:
    log_path = tmp_path / "cdbot.log"
    configure_debug_logger(level_name="DEBUG", log_path=log_path)
    client = SanitizedAppServerClient(config=AppServerConfig(codex_bin="/tmp/codex"))
    cast(Any, client)._proc = SimpleNamespace(stderr=io.StringIO("noisy\n"))

    client._start_stderr_drain_thread()
    debug_thread = client._stderr_thread
    assert debug_thread is not None
    debug_thread.join(timeout=1)

    assert "noisy" not in log_path.read_text(encoding="utf-8")

    configure_debug_logger(level_name="TRACE", log_path=log_path)
    cast(Any, client)._proc = SimpleNamespace(stderr=io.StringIO("visible\n"))

    client._start_stderr_drain_thread()
    trace_thread = client._stderr_thread
    assert trace_thread is not None
    trace_thread.join(timeout=1)

    assert "TRACE cdbot sdk.stderr visible" in log_path.read_text(encoding="utf-8")


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
    blocks: list[AssistantOutputBlock] = []

    result = await runner.run_turn(request, active_turn, lambda block: _record_block(blocks, block))

    assert not blocks
    assert result.assistant_text == "hello from item"
    assert result.final_assistant_text == "hello from item"
    assert result.token_usage_last == {
        "cached_input_tokens": 10,
        "input_tokens": 20,
        "output_tokens": 30,
        "total_tokens": 50,
    }


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


class FakeMultiMessageClient(FakeClient):
    def __init__(self) -> None:
        super().__init__(resume_error=False)
        self.notifications = [
            _assistant_notification("turn-1", "first"),
            _assistant_notification("turn-1", "second"),
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


class FakeApprovalBoundaryClient(FakeClient):
    def __init__(self, approval_handler: Any) -> None:
        super().__init__(resume_error=False)
        self._approval_handler = approval_handler
        self._approval_requested = False
        self.notifications = [
            _assistant_notification("turn-1", "first"),
            _assistant_notification("turn-1", "second"),
            _assistant_notification("turn-1", "third"),
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

    def next_turn_notification(self, turn_id: str) -> object:
        _ = turn_id
        if not self._approval_requested and len(self.notifications) == 3:
            self._approval_handler("shell.exec", {"command": "pwd"})
            self._approval_requested = True
        return self.notifications.pop(0)


@pytest.mark.asyncio
async def test_runner_emits_only_non_final_blocks() -> None:
    loop = asyncio.get_running_loop()
    runner = _runner(
        loop=loop,
        client_factory=lambda _config, _approval_handler: FakeMultiMessageClient(),
    )
    active_turn = ActiveTurn(conversation_key="dm:1")
    request = _turn_request()
    blocks: list[AssistantOutputBlock] = []

    result = await runner.run_turn(
        request,
        active_turn,
        lambda block: _record_block(blocks, block),
    )

    assert blocks == [AssistantOutputBlock(text="first", starts_new_message=False)]
    assert result.assistant_text == "firstsecond"
    assert result.final_assistant_text == "second"


@pytest.mark.asyncio
async def test_runner_marks_next_block_after_approval_as_new_message() -> None:
    loop = asyncio.get_running_loop()
    approval_router = ApprovalCaptureRouter()
    runner = _runner(
        loop=loop,
        approval_router=approval_router,
        client_factory=lambda _config, approval_handler: FakeApprovalBoundaryClient(
            approval_handler
        ),
    )
    active_turn = ActiveTurn(conversation_key="dm:1")
    request = _turn_request()
    blocks: list[AssistantOutputBlock] = []

    result = await runner.run_turn(
        request,
        active_turn,
        lambda block: _record_block(blocks, block),
    )

    assert blocks == [
        AssistantOutputBlock(text="first", starts_new_message=False),
        AssistantOutputBlock(text="second", starts_new_message=True),
    ]
    assert result.assistant_text == "firstsecondthird"
    assert result.final_assistant_text == "third"


def _assistant_notification(turn_id: str, text: str) -> FakeNotification:
    return FakeNotification(
        method="item/completed",
        payload=SimpleNamespace(
            turn_id=turn_id,
            item=SimpleNamespace(
                model_dump=lambda mode="json": {
                    "type": "agentMessage",
                    "text": text,
                }
            ),
        ),
    )


async def _ignore_delta(_: AssistantOutputBlock) -> None:
    return None


async def _record_block(
    blocks: list[AssistantOutputBlock], block: AssistantOutputBlock
) -> None:
    blocks.append(block)
