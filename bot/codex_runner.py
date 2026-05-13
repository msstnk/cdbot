"""Run Codex SDK turns and stream assistant output back to Discord."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import subprocess
import threading
from collections.abc import Callable, Coroutine
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from openai_codex import AppServerConfig, AppServerError
from openai_codex.client import AppServerClient, _resolve_codex_bin
from openai_codex.models import JsonObject, JsonValue

from .approval_router import ApprovalRequest, ApprovalRouter
from .conversation_state import ActiveTurn
from .debug_logging import dump_for_log, get_logger, trace
from .helper import parse_int
from .localization import Messages, load_default_messages
from .text import (
    extract_assistant_text,
    extract_last_assistant_text,
    extract_text_blocks_from_item,
)


class AppServerClientProtocol(Protocol):
    """Subset of the Codex app-server client used by the runner."""

    def start(self) -> None:
        """Start the underlying Codex app-server process."""

    def close(self) -> None:
        """Close the underlying Codex app-server process."""

    def initialize(self) -> Any:
        """Initialize the app-server protocol."""

    def thread_start(self, params: dict[str, Any] | None = None) -> Any:
        """Start a new Codex thread."""

    def thread_resume(self, thread_id: str, params: dict[str, Any] | None = None) -> Any:
        """Resume an existing Codex thread."""

    def turn_start(
        self,
        thread_id: str,
        input_items: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Start a turn in a Codex thread."""

    def turn_steer(self, thread_id: str, expected_turn_id: str, input_items: str) -> Any:
        """Send additional input to an active turn."""

    def turn_interrupt(self, thread_id: str, turn_id: str) -> Any:
        """Interrupt an active Codex turn."""

    def next_turn_notification(self, turn_id: str) -> Any:
        """Return the next SDK notification for a specific turn."""


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Discord request metadata for one Codex turn."""

    conversation_key: str
    requester_id: int
    channel: Any


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Session-specific Codex inputs for one turn."""

    model: str
    cwd: str
    session_id: str
    session_model: str
    session_cwd: str


@dataclass(frozen=True, slots=True)
class TurnRequest:
    """Inputs needed to run one Codex turn."""

    context: TurnContext
    prompt: str
    session: SessionContext


@dataclass(frozen=True, slots=True)
class AssistantOutputBlock:
    """One assistant output block plus Discord streaming metadata."""

    text: str
    starts_new_message: bool = False


@dataclass(frozen=True, slots=True)
class TurnResult:
    """Result metadata and assistant text from one Codex turn."""

    thread_id: str
    turn_id: str
    status: str
    assistant_text: str
    final_assistant_text: str
    resumed: bool
    token_usage_last: dict[str, int]


ApprovalHandler = Callable[[str, JsonObject | None], JsonObject]
ClientFactory = Callable[[AppServerConfig, ApprovalHandler], AppServerClientProtocol]
BlockHandler = Callable[[AssistantOutputBlock], Coroutine[Any, Any, None]]
FileChangePreview = dict[str, str]
CDBOT_ENV_PREFIX = "CDBOT_"
DEFAULT_RUNNER_DEBUG_LEVEL = "OFF"
THREAD_RESUME_FALLBACK_ERRORS = (AppServerError, OSError, RuntimeError, ValueError)
APPROVAL_REQUEST_ERRORS = (
    RuntimeError,
    concurrent.futures.CancelledError,
    concurrent.futures.TimeoutError,
)


@dataclass(frozen=True, slots=True)
class RunnerEnvironment:
    """Static Codex app-server configuration shared by all turns."""

    codex_bin: str
    codex_home: str | Path
    workspace_cwd: str
    debug_level_name: str = DEFAULT_RUNNER_DEBUG_LEVEL


@dataclass(frozen=True, slots=True)
class RunnerDependencies:
    """Runtime collaborators used by the Codex runner."""

    approval_router: ApprovalRouter
    loop: asyncio.AbstractEventLoop
    client_factory: ClientFactory | None = None
    messages: Messages | None = None


@dataclass(frozen=True, slots=True)
class TurnRuntime:
    """Turn collaborators wrapped for helper methods."""

    request: TurnRequest
    active_turn: ActiveTurn
    on_block: BlockHandler


@dataclass(slots=True)
class TurnOutputState:
    """Assistant output accumulated while one Codex turn is running."""

    assistant_parts: list[str] = field(default_factory=list)
    pending_output_block: AssistantOutputBlock | None = None
    next_output_starts_new_message: bool = False
    completed_turn: object | None = None


@dataclass(slots=True)
class TurnExecutionState:
    """State accumulated while one Codex turn is running."""

    file_change_preview_by_item_id: dict[str, list[FileChangePreview]] = field(
        default_factory=dict
    )
    output: TurnOutputState = field(default_factory=TurnOutputState)
    token_usage_last: dict[str, int] = field(default_factory=dict)
    resumed: bool = False
    status: str = "unknown"
    thread_id: str = ""
    turn_id: str = ""


class SanitizedAppServerClient(AppServerClient):
    """App-server client that keeps bot-only environment variables private."""

    def __init__(
        self,
        config: AppServerConfig | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        super().__init__(config=config, approval_handler=approval_handler)
        self._proc_context: ExitStack | None = None

    def start(self) -> None:
        if self._proc is not None:
            return

        if self.config.launch_args_override is not None:
            args = list(self.config.launch_args_override)
        else:
            codex_bin = _resolve_codex_bin(self.config)
            args = [str(codex_bin)]
            for kv in self.config.config_overrides:
                args.extend(["--config", kv])
            args.extend(["app-server", "--listen", "stdio://"])

        proc_context = ExitStack()
        self._proc = proc_context.enter_context(
            subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=self.config.cwd,
                env=_build_app_server_env(self.config.env),
                bufsize=1,
            )
        )
        self._proc_context = proc_context
        self._start_stderr_drain_thread()
        self._start_reader_thread()

    def close(self) -> None:
        proc_context = cast(ExitStack | None, getattr(self, "_proc_context", None))
        self._proc_context = None
        super().close()
        if proc_context is not None:
            proc_context.close()

    def _start_stderr_drain_thread(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return

        proc = self._proc
        logger = get_logger()

        def _drain() -> None:
            stderr = proc.stderr
            if stderr is None:
                return
            for line in stderr:
                message = line.rstrip("\n")
                self._stderr_lines.append(message)
                trace(logger, "sdk.stderr %s", message)

        self._stderr_thread = threading.Thread(target=_drain, daemon=True)
        self._stderr_thread.start()


class CodexRunner:
    """Synchronous Codex SDK adapter with an async public interface."""

    def __init__(
        self,
        *,
        environment: RunnerEnvironment,
        dependencies: RunnerDependencies,
    ) -> None:
        self._approval_router = dependencies.approval_router
        self._logger = get_logger()
        self._loop = dependencies.loop
        self._environment = RunnerEnvironment(
            codex_bin=environment.codex_bin,
            codex_home=str(environment.codex_home),
            workspace_cwd=environment.workspace_cwd,
            debug_level_name=environment.debug_level_name,
        )
        self._client_factory = dependencies.client_factory or self._default_client_factory
        self._messages = dependencies.messages or load_default_messages()

    async def run_turn(
        self,
        request: TurnRequest,
        active_turn: ActiveTurn,
        on_block: BlockHandler,
    ) -> TurnResult:
        """Run one turn without blocking the Discord event loop."""
        return await asyncio.to_thread(self.run_turn_sync, request, active_turn, on_block)

    def run_turn_sync(
        self,
        request: TurnRequest,
        active_turn: ActiveTurn,
        on_block: BlockHandler,
    ) -> TurnResult:
        """Run one turn on the calling thread."""
        runtime = TurnRuntime(
            request=request,
            active_turn=active_turn,
            on_block=on_block,
        )
        state = TurnExecutionState()
        config = self._build_config(request.session.cwd)
        self._log_turn_start(runtime.request)
        client = self._client_factory(
            config,
            self._make_approval_handler(
                runtime.request,
                runtime.active_turn,
                state,
                state.file_change_preview_by_item_id,
            ),
        )

        try:
            self._start_client(client, config, runtime.request)
            self._start_or_resume_thread(client, runtime.request, state)
            self._start_turn(client, runtime, state)
            self._stream_notifications(client, runtime, state)
        finally:
            self._logger.debug(
                "sdk.close conversation_key=%s thread_id=%s turn_id=%s",
                runtime.request.context.conversation_key,
                state.thread_id or "-",
                state.turn_id or "-",
            )
            self._approval_router.clear_turn_approval(
                runtime.request.context.conversation_key,
                state.turn_id,
            )
            client.close()

        assistant_text = (
            "".join(state.output.assistant_parts)
            or extract_assistant_text(state.output.completed_turn)
        )
        final_assistant_text = (
            state.output.pending_output_block.text
            if state.output.pending_output_block is not None
            else extract_last_assistant_text(state.output.completed_turn)
        )
        self._logger.debug(
            (
                "turn.run.result conversation_key=%s thread_id=%s "
                "turn_id=%s resumed=%s assistant_chars=%s"
            ),
            runtime.request.context.conversation_key,
            state.thread_id,
            state.turn_id,
            state.resumed,
            len(assistant_text),
        )
        return TurnResult(
            thread_id=state.thread_id,
            turn_id=state.turn_id,
            status=state.status,
            assistant_text=assistant_text,
            final_assistant_text=final_assistant_text,
            resumed=state.resumed,
            token_usage_last=dict(state.token_usage_last),
        )

    def _log_turn_start(self, request: TurnRequest) -> None:
        self._logger.info(
            "turn.run.start conversation_key=%s model=%s cwd=%s session_id=%s",
            request.context.conversation_key,
            request.session.model,
            request.session.cwd,
            request.session.session_id or "-",
        )

    def _start_client(
        self,
        client: AppServerClientProtocol,
        config: AppServerConfig,
        request: TurnRequest,
    ) -> None:
        self._logger.debug("sdk.start config=%s", dump_for_log(config))
        self._logger.debug(
            "sdk.process.start conversation_key=%s",
            request.context.conversation_key,
        )
        client.start()
        self._logger.debug(
            "sdk.process.started conversation_key=%s pid=%s",
            request.context.conversation_key,
            _client_pid(client),
        )
        self._logger.debug(
            "sdk.initialize.start conversation_key=%s",
            request.context.conversation_key,
        )
        client.initialize()
        self._logger.debug(
            "sdk.initialize.ok conversation_key=%s",
            request.context.conversation_key,
        )

    def _start_or_resume_thread(
        self,
        client: AppServerClientProtocol,
        request: TurnRequest,
        state: TurnExecutionState,
    ) -> None:
        should_resume = self._should_resume_thread(request)
        if should_resume:
            self._resume_thread(client, request, state)
            return
        self._start_new_thread(client, request, state)

    def _should_resume_thread(self, request: TurnRequest) -> bool:
        should_resume = bool(request.session.session_id)
        if (
            should_resume
            and request.session.session_model
            and request.session.model != request.session.session_model
        ):
            self._logger.info(
                (
                    "sdk.thread_resume.skipped thread_id=%s "
                    "reason=model_changed old_model=%s new_model=%s"
                ),
                request.session.session_id,
                request.session.session_model,
                request.session.model,
            )
            return False
        return should_resume

    def _resume_thread(
        self,
        client: AppServerClientProtocol,
        request: TurnRequest,
        state: TurnExecutionState,
    ) -> None:
        try:
            self._logger.debug(
                "sdk.thread_resume.request thread_id=%s params=%s",
                request.session.session_id,
                dump_for_log({"model": request.session.model}),
            )
            resumed_response = client.thread_resume(
                request.session.session_id,
                {"model": request.session.model},
            )
            state.thread_id = cast(str, resumed_response.thread.id)
            state.resumed = True
            self._logger.info("sdk.thread_resume.ok thread_id=%s", state.thread_id)
        except THREAD_RESUME_FALLBACK_ERRORS:
            self._logger.exception(
                "sdk.thread_resume.failed thread_id=%s; falling back to thread_start",
                request.session.session_id,
            )
            self._start_new_thread(client, request, state)

    def _start_new_thread(
        self,
        client: AppServerClientProtocol,
        request: TurnRequest,
        state: TurnExecutionState,
    ) -> None:
        self._logger.debug(
            "sdk.thread_start.request params=%s",
            dump_for_log({"model": request.session.model}),
        )
        started_response = client.thread_start({"model": request.session.model})
        state.thread_id = cast(str, started_response.thread.id)
        self._logger.info("sdk.thread_start.ok thread_id=%s", state.thread_id)

    def _start_turn(
        self,
        client: AppServerClientProtocol,
        runtime: TurnRuntime,
        state: TurnExecutionState,
    ) -> None:
        turn_input = runtime.request.prompt
        if (
            not state.resumed
            or runtime.request.session.cwd != runtime.request.session.session_cwd
        ):
            turn_input = self._messages.format(
                "codex.initial_turn_input",
                cwd=runtime.request.session.cwd,
                prompt=runtime.request.prompt,
            )
        self._logger.debug(
            "sdk.turn_start.request thread_id=%s params=%s input=%s",
            state.thread_id,
            dump_for_log({"model": runtime.request.session.model}),
            dump_for_log(turn_input),
        )
        turn_response = client.turn_start(
            state.thread_id,
            turn_input,
            params={"model": runtime.request.session.model},
        )
        state.turn_id = cast(str, turn_response.turn.id)
        self._logger.info(
            "sdk.turn_start.ok thread_id=%s turn_id=%s",
            state.thread_id,
            state.turn_id,
        )
        runtime.active_turn.ids.thread_id = state.thread_id
        runtime.active_turn.ids.turn_id = state.turn_id

    def _stream_notifications(
        self,
        client: AppServerClientProtocol,
        runtime: TurnRuntime,
        state: TurnExecutionState,
    ) -> None:
        while True:
            self._flush_turn_controls(client, runtime.active_turn, state)
            notification = client.next_turn_notification(state.turn_id)
            if self._handle_notification(notification, runtime, state):
                return

    def _flush_turn_controls(
        self,
        client: AppServerClientProtocol,
        active_turn: ActiveTurn,
        state: TurnExecutionState,
    ) -> None:
        if active_turn.take_interrupt_request():
            self._logger.debug(
                "sdk.turn_interrupt.request thread_id=%s turn_id=%s",
                state.thread_id,
                state.turn_id,
            )
            client.turn_interrupt(state.thread_id, state.turn_id)

        steer_text = active_turn.drain_steer()
        if steer_text:
            self._logger.debug(
                "sdk.turn_steer.request thread_id=%s turn_id=%s input=%s",
                state.thread_id,
                state.turn_id,
                dump_for_log(steer_text),
            )
            client.turn_steer(state.thread_id, state.turn_id, steer_text)

    def _handle_notification(
        self,
        notification: object,
        runtime: TurnRuntime,
        state: TurnExecutionState,
    ) -> bool:
        method = getattr(notification, "method", "")
        payload = getattr(notification, "payload", None)
        trace(
            self._logger,
            "sdk.notification method=%s payload=%s",
            method,
            dump_for_log(payload),
        )

        if method == "thread/tokenUsage/updated":
            token_usage_last = _extract_token_usage_last(payload)
            if token_usage_last:
                state.token_usage_last = token_usage_last
            return False

        if method == "item/started" and getattr(payload, "turn_id", "") == state.turn_id:
            item = getattr(payload, "item", None)
            item_id = _extract_thread_item_id(item)
            if _extract_thread_item_type(item) == "fileChange" and item_id:
                state.file_change_preview_by_item_id[item_id] = _extract_file_change_preview(
                    item
                )
            return False

        if method == "item/completed" and getattr(payload, "turn_id", "") == state.turn_id:
            self._handle_completed_item(payload, runtime, state)
            return False

        if method == "turn/completed" and getattr(
            getattr(payload, "turn", None), "id", ""
        ) == state.turn_id:
            completed_payload = cast(Any, payload)
            state.output.completed_turn = completed_payload.turn
            raw_status = getattr(
                completed_payload.turn.status,
                "value",
                completed_payload.turn.status,
            )
            state.status = str(raw_status)
            self._logger.info(
                "sdk.turn_completed thread_id=%s turn_id=%s status=%s",
                state.thread_id,
                state.turn_id,
                state.status,
            )
            return True

        return False

    def _handle_completed_item(
        self,
        payload: object,
        runtime: TurnRuntime,
        state: TurnExecutionState,
    ) -> None:
        text = "".join(extract_text_blocks_from_item(getattr(payload, "item", None)))
        if not text:
            return

        state.output.assistant_parts.append(text)
        completed_block = AssistantOutputBlock(
            text=text,
            starts_new_message=state.output.next_output_starts_new_message,
        )
        state.output.next_output_starts_new_message = False

        if (
            state.output.pending_output_block is not None
            and not runtime.active_turn.discard_output
        ):
            self._run_coroutine(runtime.on_block(state.output.pending_output_block))
        state.output.pending_output_block = completed_block

    def _make_approval_handler(
        self,
        request: TurnRequest,
        active_turn: ActiveTurn,
        state: TurnExecutionState,
        file_change_preview_by_item_id: dict[str, list[FileChangePreview]],
    ) -> ApprovalHandler:
        def approval_handler(method: str, params: JsonObject | None) -> JsonObject:
            state.output.next_output_starts_new_message = True
            approval_params = dict(params or {})
            if method == "item/fileChange/requestApproval":
                approval_item_id = _extract_file_change_item_id(approval_params)
                previews = file_change_preview_by_item_id.get(approval_item_id, [])
                if previews:
                    approval_params["fileChangeFiles"] = [
                        entry["path"] for entry in previews if entry.get("path")
                    ]
                    approval_params["fileChangeChanges"] = cast(list[JsonValue], previews)
            self._logger.debug(
                "sdk.approval.request conversation_key=%s method=%s params=%s",
                request.context.conversation_key,
                method,
                dump_for_log(approval_params),
            )
            future = asyncio.run_coroutine_threadsafe(
                self._approval_router.request_approval(
                    ApprovalRequest(
                        channel=request.context.channel,
                        requester_id=request.context.requester_id,
                        conversation_key=request.context.conversation_key,
                        turn_id=active_turn.ids.turn_id,
                        method=method,
                        params=approval_params,
                    )
                ),
                self._loop,
            )
            try:
                result = cast(JsonObject, future.result())
                self._logger.debug(
                    "sdk.approval.result conversation_key=%s method=%s result=%s",
                    request.context.conversation_key,
                    method,
                    dump_for_log(result),
                )
                return result
            except APPROVAL_REQUEST_ERRORS:
                self._logger.exception(
                    "sdk.approval.failed conversation_key=%s method=%s",
                    request.context.conversation_key,
                    method,
                )
                return cast(JsonObject, {"decision": "deny"})

        return approval_handler

    def _build_config(self, cwd: str) -> AppServerConfig:
        return AppServerConfig(
            codex_bin=self._environment.codex_bin,
            cwd=cwd or self._environment.workspace_cwd,
            env=_build_codex_runtime_env(
                codex_home=self._environment.codex_home,
                debug_level_name=self._environment.debug_level_name,
            ),
        )

    def _run_coroutine(self, awaitable: Coroutine[Any, Any, None]) -> None:
        future: concurrent.futures.Future[None] = asyncio.run_coroutine_threadsafe(
            awaitable,
            self._loop,
        )
        future.result()

    @staticmethod
    def _default_client_factory(
        config: AppServerConfig,
        approval_handler: ApprovalHandler,
    ) -> AppServerClientProtocol:
        return SanitizedAppServerClient(config=config, approval_handler=approval_handler)


def _build_app_server_env(env_overrides: dict[str, str] | None) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(CDBOT_ENV_PREFIX)
    }
    if env_overrides:
        env.update(env_overrides)
    return env


def _build_codex_runtime_env(*, codex_home: str | Path, debug_level_name: str) -> dict[str, str]:
    env = {"CODEX_HOME": str(codex_home)}
    rust_log = _child_rust_log_level(debug_level_name)
    if rust_log:
        env["RUST_LOG"] = rust_log
    return env


def _child_rust_log_level(level_name: str) -> str:
    normalized = level_name.strip().upper()
    if normalized in {"OFF", "NONE"}:
        return ""
    if normalized == "WARNING":
        return "warn"
    return normalized.lower()


def _client_pid(client: object) -> str:
    proc = getattr(client, "_proc", None)
    pid = getattr(proc, "pid", None)
    return str(pid or "-")


def _extract_file_change_item_id(payload: JsonObject) -> str:
    direct = str(payload.get("itemId") or payload.get("item_id") or "").strip()
    if direct:
        return direct
    item = payload.get("item")
    if isinstance(item, dict):
        return str(item.get("id") or "").strip()
    return ""


def _extract_file_change_preview(item: object) -> list[FileChangePreview]:
    raw_item = _thread_item_payload(item)
    if not isinstance(raw_item, dict):
        return []
    changes = raw_item.get("changes")
    if not isinstance(changes, list):
        return []

    preview: list[FileChangePreview] = []
    for change in changes[:5]:
        if not isinstance(change, dict):
            continue
        path = str(change.get("path") or "").strip()
        if not path:
            continue
        kind = _extract_patch_change_kind(change)
        snippet = str(change.get("diff") or "").strip()
        preview.append(
            {
                "path": path,
                "kind": kind or "-",
                "snippet": snippet,
            }
        )
    return preview


def _extract_patch_change_kind(change: dict[str, Any]) -> str:
    kind_payload = change.get("kind")
    if isinstance(kind_payload, dict):
        return str(kind_payload.get("type") or "").strip()
    return str(change.get("type") or kind_payload or "").strip()


def _extract_thread_item_id(item: object) -> str:
    raw_item = _thread_item_payload(item)
    if not isinstance(raw_item, dict):
        return ""
    return str(raw_item.get("id") or "").strip()


def _extract_thread_item_type(item: object) -> str:
    raw_item = _thread_item_payload(item)
    if not isinstance(raw_item, dict):
        return ""
    return str(raw_item.get("type") or "").strip()


def _thread_item_payload(item: object) -> object:
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return item


def _extract_token_usage_last(payload: object) -> dict[str, int]:
    token_usage = _payload_value(payload, "token_usage", "tokenUsage")
    total = _payload_value(token_usage, "total")
    return _normalize_token_usage(total)


def _normalize_token_usage(usage_last: object) -> dict[str, int]:
    cached_input = _int_payload_value(usage_last, "cached_input_tokens", "cachedInputTokens")
    input_tokens = _int_payload_value(usage_last, "input_tokens", "inputTokens")
    output_tokens = _int_payload_value(usage_last, "output_tokens", "outputTokens")
    total_tokens = _int_payload_value(usage_last, "total_tokens", "totalTokens")
    if total_tokens <= 0:
        total_tokens = max(input_tokens, cached_input) + output_tokens
    if cached_input <= 0 and input_tokens <= 0 and output_tokens <= 0 and total_tokens <= 0:
        return {}
    return {
        "cached_input_tokens": cached_input,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _int_payload_value(payload: object, *keys: str) -> int:
    value = _payload_value(payload, *keys)
    if value is None:
        return 0
    return parse_int(value)


def _payload_value(payload: object, *keys: str) -> object:
    if payload is None:
        return None
    for key in keys:
        if isinstance(payload, dict) and key in payload:
            return payload[key]
        attr = getattr(payload, key, None)
        if attr is not None:
            return attr
    return None
