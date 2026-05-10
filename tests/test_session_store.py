from __future__ import annotations

from pathlib import Path

from bot.session_store import SessionStore


def test_session_store_replays_latest_records(tmp_path: Path) -> None:
    path = tmp_path / "sessions.jsonl"
    store = SessionStore(path)

    store.set_discord_user_id("dm:1", 42)
    store.save_session(
        "dm:1",
        session_id="thread-1",
        session_model="gpt-5.4",
        session_cwd="/tmp/workspace",
    )
    store.set_model("dm:1", "gpt-5.5")
    store.set_cwd("dm:1", "/tmp/workspace")
    reloaded = SessionStore(path)

    assert reloaded.get_session_id("dm:1") == "thread-1"
    assert reloaded.get_session_model("dm:1") == "gpt-5.4"
    assert reloaded.get_discord_user_id("dm:1") == 42
    assert reloaded.get_model("dm:1", "fallback") == "gpt-5.5"
    assert reloaded.get_cwd("dm:1", "fallback") == "/tmp/workspace"
    assert reloaded.get_session_cwd("dm:1") == "/tmp/workspace"


def test_session_store_save_keeps_one_row_per_conversation(tmp_path: Path) -> None:
    path = tmp_path / "sessions.jsonl"
    store = SessionStore(path)

    store.save_session(
        "dm:1",
        session_id="thread-1",
        session_model="gpt-5.4",
        session_cwd="/tmp/workspace",
    )
    store.set_model("dm:1", "gpt-5.4")
    store.set_cwd("dm:1", "/tmp/workspace")
    store.set_model("dm:1", "gpt-5.5")

    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1
    assert lines[0].count('"conversation_key": "dm:1"') == 1


def test_session_store_clear_deletes_record(tmp_path: Path) -> None:
    path = tmp_path / "sessions.jsonl"
    store = SessionStore(path)

    store.set_discord_user_id("dm:1", 42)
    store.save_session(
        "dm:1",
        session_id="thread-1",
        session_model="gpt-5.4",
        session_cwd="/tmp/workspace",
    )
    store.set_model("dm:1", "gpt-5.4")
    store.set_cwd("dm:1", "/tmp/workspace")
    store.clear("dm:1")
    reloaded = SessionStore(path)

    assert reloaded.get_session_id("dm:1") == ""
    assert reloaded.get_session_model("dm:1") == ""
    assert reloaded.get_session_cwd("dm:1") == ""
    assert reloaded.get_discord_user_id("dm:1") == 42
    assert reloaded.get_model("dm:1", "fallback") == "gpt-5.4"
    assert reloaded.get_cwd("dm:1", "fallback") == "/tmp/workspace"
    assert path.exists()


def test_session_store_set_cwd_preserves_model_and_session(tmp_path: Path) -> None:
    path = tmp_path / "sessions.jsonl"
    store = SessionStore(path)

    store.save_session(
        "dm:1",
        session_id="thread-1",
        session_model="gpt-5.4",
        session_cwd="/tmp/original",
    )
    store.set_model("dm:1", "gpt-5.4")
    store.set_cwd("dm:1", "/tmp/original")
    store.set_cwd("dm:1", "/tmp/updated")
    reloaded = SessionStore(path)

    assert reloaded.get_session_id("dm:1") == "thread-1"
    assert reloaded.get_session_model("dm:1") == "gpt-5.4"
    assert reloaded.get_model("dm:1", "fallback") == "gpt-5.4"
    assert reloaded.get_cwd("dm:1", "fallback") == "/tmp/updated"
    assert reloaded.get_session_cwd("dm:1") == "/tmp/original"


def test_session_store_set_discord_user_id_keeps_defaults_and_session(tmp_path: Path) -> None:
    path = tmp_path / "sessions.jsonl"
    store = SessionStore(path)

    store.save_session(
        "dm:1",
        session_id="thread-1",
        session_model="gpt-5.4",
        session_cwd="/tmp/workspace",
    )
    store.set_model("dm:1", "gpt-5.4")
    store.set_cwd("dm:1", "/tmp/workspace")
    store.set_discord_user_id("dm:1", 42)
    reloaded = SessionStore(path)

    assert reloaded.get_session_id("dm:1") == "thread-1"
    assert reloaded.get_session_model("dm:1") == "gpt-5.4"
    assert reloaded.get_session_cwd("dm:1") == "/tmp/workspace"
    assert reloaded.get_model("dm:1", "fallback") == "gpt-5.4"
    assert reloaded.get_cwd("dm:1", "fallback") == "/tmp/workspace"
    assert reloaded.get_discord_user_id("dm:1") == 42


def test_session_store_clear_deletes_empty_record(tmp_path: Path) -> None:
    path = tmp_path / "sessions.jsonl"
    store = SessionStore(path)

    store.save_session(
        "dm:1",
        session_id="thread-1",
        session_model="gpt-5.4",
        session_cwd="/tmp/workspace",
    )
    store.clear("dm:1")

    assert not path.exists()
