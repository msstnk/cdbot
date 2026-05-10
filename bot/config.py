"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .localization import DEFAULT_LOCALE, DEFAULT_LOCALES_PATH

DEFAULT_CODEX_BIN = Path(".codex/bin/codex")
DEFAULT_CODEX_HOME = Path(".codex")
DEFAULT_SESSION_STORE = Path(".local/session_store.jsonl")
DEFAULT_DEBUG_LOG_PATH = Path(".local/cdbot.log")
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_APPROVAL_TIMEOUT_SEC = 60.0
DEFAULT_CDBOT_DEBUG_LEVEL = "OFF"
DEFAULT_OPENAI_TRANSCRIPTION_MODEL = "whisper-1"
DEFAULT_ENABLE_VOICE_CONTROL = False
DISCORD_MESSAGE_LIMIT = 2000
ENV_DISCORD_BOT_TOKEN = "CDBOT_DISCORD_BOT_TOKEN"
ENV_CODEX_BIN = "CDBOT_CODEX_BIN"
ENV_CODEX_HOME = "CDBOT_CODEX_HOME"
ENV_CODEX_MODEL = "CDBOT_CODEX_MODEL"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_TRANSCRIPTION_MODEL = "CDBOT_OPENAI_TRANSCRIPTION_MODEL"
ENV_ENABLE_VOICE_CONTROL = "CDBOT_ENABLE_VOICE_CONTROL"
ENV_APPROVAL_TIMEOUT_SEC = "CDBOT_APPROVAL_TIMEOUT_SEC"
ENV_SESSION_STORE_PATH = "CDBOT_SESSION_STORE_PATH"
ENV_WORKSPACE_CWD = "CDBOT_WORKSPACE_CWD"
ENV_DEBUG_LEVEL = "CDBOT_DEBUG_LEVEL"
ENV_DEBUG_LOG_PATH = "CDBOT_DEBUG_LOG_PATH"
ENV_LOCALE = "CDBOT_LOCALE"
ENV_WHITELISTED_USERS = "CDBOT_WHITELISTED_USERS"


@dataclass(frozen=True, slots=True)
class CodexSettings:
    """Codex runtime settings shared across Discord bot components."""

    bin_path: str
    home_path: Path
    default_model: str
    workspace_cwd: str


@dataclass(frozen=True, slots=True)
class OpenAISettings:
    """OpenAI settings used for optional voice transcription."""

    enable_voice_control: bool
    api_key: str
    transcription_model: str


@dataclass(frozen=True, slots=True)
class StorageSettings:
    """Filesystem locations used by the Discord bot."""

    session_store_path: Path
    debug_log_path: Path


@dataclass(frozen=True, slots=True)
class DebugSettings:
    """Debug logging configuration."""

    level_name: str


@dataclass(frozen=True, slots=True)
class LocaleSettings:
    """Localization catalog configuration."""

    locale: str = DEFAULT_LOCALE
    path: Path = DEFAULT_LOCALES_PATH.resolve()


@dataclass(frozen=True, slots=True)
# pylint: disable=too-many-instance-attributes
class Settings:
    """Runtime settings for the Discord bot and Codex runner."""

    discord_bot_token: str
    whitelisted_users: frozenset[int]
    codex: CodexSettings
    openai: OpenAISettings
    approval_timeout_sec: float
    storage: StorageSettings
    debug: DebugSettings
    localization: LocaleSettings

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables and validated defaults."""
        token = os.environ.get(ENV_DISCORD_BOT_TOKEN, "").strip()
        if not token:
            raise RuntimeError(f"{ENV_DISCORD_BOT_TOKEN} is required")

        codex_bin = os.environ.get(ENV_CODEX_BIN, "").strip()
        if not codex_bin:
            default_codex_bin = DEFAULT_CODEX_BIN.resolve()
            codex_bin = str(default_codex_bin)
        if not Path(codex_bin).exists():
            raise RuntimeError(f"Codex binary was not found: {codex_bin}")

        approval_timeout_raw = os.environ.get(ENV_APPROVAL_TIMEOUT_SEC, "").strip()
        approval_timeout_sec = DEFAULT_APPROVAL_TIMEOUT_SEC
        if approval_timeout_raw:
            approval_timeout_sec = max(1.0, float(approval_timeout_raw))

        session_store_path = Path(
            os.environ.get(ENV_SESSION_STORE_PATH, str(DEFAULT_SESSION_STORE))
        ).resolve()
        codex_home = Path(
            os.environ.get(ENV_CODEX_HOME, str(DEFAULT_CODEX_HOME))
        ).expanduser().resolve()
        workspace_cwd = str(
            Path(os.environ.get(ENV_WORKSPACE_CWD, os.getcwd())).expanduser().resolve()
        )
        cdbot_debug_level = (
            os.environ.get(ENV_DEBUG_LEVEL, DEFAULT_CDBOT_DEBUG_LEVEL).strip().upper()
            or DEFAULT_CDBOT_DEBUG_LEVEL
        )
        debug_log_path = Path(
            os.environ.get(ENV_DEBUG_LOG_PATH, str(DEFAULT_DEBUG_LOG_PATH))
        ).resolve()
        locales_path = DEFAULT_LOCALES_PATH.resolve()
        if not locales_path.exists():
            raise RuntimeError(f"Locales file was not found: {locales_path}")
        locale = os.environ.get(ENV_LOCALE, DEFAULT_LOCALE).strip() or DEFAULT_LOCALE
        whitelisted_users = _parse_whitelisted_users(
            os.environ.get(ENV_WHITELISTED_USERS, "")
        )

        return cls(
            discord_bot_token=token,
            whitelisted_users=whitelisted_users,
            codex=CodexSettings(
                bin_path=codex_bin,
                home_path=codex_home,
                default_model=(
                    os.environ.get(ENV_CODEX_MODEL, DEFAULT_MODEL).strip() or DEFAULT_MODEL
                ),
                workspace_cwd=workspace_cwd,
            ),
            openai=OpenAISettings(
                enable_voice_control=_parse_bool_env(
                    raw_value=os.environ.get(ENV_ENABLE_VOICE_CONTROL, ""),
                    default=DEFAULT_ENABLE_VOICE_CONTROL,
                    env_name=ENV_ENABLE_VOICE_CONTROL,
                ),
                api_key=os.environ.get(ENV_OPENAI_API_KEY, "").strip(),
                transcription_model=(
                    os.environ.get(
                        ENV_OPENAI_TRANSCRIPTION_MODEL,
                        DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
                    ).strip()
                    or DEFAULT_OPENAI_TRANSCRIPTION_MODEL
                ),
            ),
            approval_timeout_sec=approval_timeout_sec,
            storage=StorageSettings(
                session_store_path=session_store_path,
                debug_log_path=debug_log_path,
            ),
            debug=DebugSettings(level_name=cdbot_debug_level),
            localization=LocaleSettings(locale=locale, path=locales_path),
        )


def _parse_whitelisted_users(raw_value: str) -> frozenset[int]:
    normalized = raw_value.strip()
    if not normalized:
        return frozenset()

    user_ids: set[int] = set()
    for part in normalized.split(","):
        candidate = part.strip()
        if not candidate:
            continue
        try:
            user_ids.add(int(candidate))
        except ValueError as exc:
            raise RuntimeError(
                f"{ENV_WHITELISTED_USERS} must be a comma-separated list of Discord user ids"
            ) from exc
    return frozenset(user_ids)


def _parse_bool_env(*, raw_value: str, default: bool, env_name: str) -> bool:
    normalized = raw_value.strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{env_name} must be one of true/false, 1/0, yes/no, on/off")
