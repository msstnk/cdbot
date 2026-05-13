"""Persist Discord conversation session metadata as JSON Lines."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .helper import now_iso


@dataclass(frozen=True, slots=True)
class SessionIdentity:
    """Stored Codex thread identity for one conversation."""

    session_id: str = ""
    session_model: str = ""
    session_cwd: str = ""


@dataclass(frozen=True, slots=True)
class ConversationDefaults:
    """Stored user-selected defaults for one conversation."""

    model: str = ""
    cwd: str = ""


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """Stored state for one Discord conversation."""

    conversation_key: str
    discord_user_id: int
    identity: SessionIdentity
    defaults: ConversationDefaults
    created_at: str
    updated_at: str


class SessionStore:
    """Lazy-loading JSON Lines store for Codex session state."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._records: dict[str, SessionRecord] = {}
        self._loaded = False

    def get(self, conversation_key: str) -> SessionRecord | None:
        """Return the stored record for a conversation, if present."""
        self._load_if_needed()
        return self._records.get(conversation_key)

    def get_session_id(self, conversation_key: str) -> str:
        """Return the Codex thread id for a conversation."""
        record = self.get(conversation_key)
        if record is None:
            return ""
        return record.identity.session_id

    def get_session_cwd(self, conversation_key: str) -> str:
        """Return the cwd associated with the stored Codex thread."""
        record = self.get(conversation_key)
        if record is None:
            return ""
        return record.identity.session_cwd

    def get_session_model(self, conversation_key: str) -> str:
        """Return the model associated with the stored Codex thread."""
        record = self.get(conversation_key)
        if record is None:
            return ""
        return record.identity.session_model

    def get_discord_user_id(self, conversation_key: str) -> int:
        """Return the Discord user id associated with a conversation."""
        record = self.get(conversation_key)
        if record is None:
            return 0
        return record.discord_user_id

    def get_model(self, conversation_key: str, default_model: str) -> str:
        """Return the saved model override or the configured default."""
        record = self.get(conversation_key)
        if record is None or not record.defaults.model:
            return default_model
        return record.defaults.model

    def get_cwd(self, conversation_key: str, default_cwd: str) -> str:
        """Return the saved cwd override or the configured default."""
        record = self.get(conversation_key)
        if record is None or not record.defaults.cwd:
            return default_cwd
        return record.defaults.cwd

    def save_session(
        self,
        conversation_key: str,
        *,
        session_id: str,
        session_model: str,
        session_cwd: str,
    ) -> SessionRecord:
        """Save Codex thread information for a conversation."""
        return self._save_record(
            conversation_key,
            identity=SessionIdentity(
                session_id=session_id,
                session_model=session_model,
                session_cwd=session_cwd,
            ),
        )

    def set_model(self, conversation_key: str, model: str) -> SessionRecord:
        """Persist the model override for a conversation."""
        record = self.get(conversation_key)
        cwd = "" if record is None else record.defaults.cwd
        return self._save_record(
            conversation_key,
            defaults=ConversationDefaults(model=model, cwd=cwd),
        )

    def set_cwd(self, conversation_key: str, cwd: str) -> SessionRecord:
        """Persist the cwd override for a conversation."""
        record = self.get(conversation_key)
        model = "" if record is None else record.defaults.model
        return self._save_record(
            conversation_key,
            defaults=ConversationDefaults(model=model, cwd=cwd),
        )

    def set_discord_user_id(self, conversation_key: str, discord_user_id: int) -> SessionRecord:
        """Persist the Discord user id for a conversation."""
        return self._save_record(
            conversation_key,
            discord_user_id=discord_user_id,
        )

    def clear(self, conversation_key: str) -> None:
        """Clear session identity while preserving stored DM metadata."""
        self._load_if_needed()
        existing = self._records.get(conversation_key)
        if existing is None:
            return
        if existing.discord_user_id or existing.defaults.model or existing.defaults.cwd:
            self._records[conversation_key] = replace(
                existing,
                identity=SessionIdentity(),
                updated_at=now_iso(),
            )
        else:
            del self._records[conversation_key]
        self._write_records()

    def _save_record(
        self,
        conversation_key: str,
        *,
        identity: SessionIdentity | None = None,
        defaults: ConversationDefaults | None = None,
        discord_user_id: int | None = None,
    ) -> SessionRecord:
        self._load_if_needed()
        now = now_iso()
        existing = self._records.get(conversation_key)
        if existing is None:
            record = SessionRecord(
                conversation_key=conversation_key,
                discord_user_id=0,
                identity=SessionIdentity(),
                defaults=ConversationDefaults(),
                created_at=now,
                updated_at=now,
            )
        else:
            record = replace(existing, updated_at=now)

        if identity is not None:
            record = replace(record, identity=identity)
        if defaults is not None:
            record = replace(record, defaults=defaults)
        if discord_user_id is not None:
            record = replace(record, discord_user_id=discord_user_id)

        self._records[conversation_key] = record
        self._write_records()
        return record

    def _load_if_needed(self) -> None:
        if self._loaded:
            return
        self._records.clear()
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                conversation_key = (raw.get("conversation_key") or "").strip()
                if not conversation_key:
                    continue
                updated_at = raw.get("updated_at") or now_iso()
                self._records[conversation_key] = SessionRecord(
                    conversation_key=conversation_key,
                    discord_user_id=raw.get("discord_user_id") or 0,
                    identity=SessionIdentity(
                        session_id=raw.get("session_id") or "",
                        session_model=raw.get("session_model") or "",
                        session_cwd=raw.get("session_cwd") or "",
                    ),
                    defaults=ConversationDefaults(
                        model=raw.get("model") or "",
                        cwd=raw.get("cwd") or "",
                    ),
                    created_at=raw.get("created_at") or updated_at,
                    updated_at=updated_at,
                )
        self._loaded = True

    def _write_records(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._records:
            self._path.unlink(missing_ok=True)
            return
        with self._path.open("w", encoding="utf-8") as handle:
            for conversation_key in sorted(self._records):
                record = self._records[conversation_key]
                handle.write(
                    json.dumps(
                        {
                            "conversation_key": record.conversation_key,
                            "discord_user_id": record.discord_user_id,
                            "session_id": record.identity.session_id,
                            "session_model": record.identity.session_model,
                            "session_cwd": record.identity.session_cwd,
                            "created_at": record.created_at,
                            "updated_at": record.updated_at,
                            "model": record.defaults.model,
                            "cwd": record.defaults.cwd,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
