"""delegation_threads + continue ownership checks."""

import json
import time

import pytest

from tools.delegation_threads import (
    ensure_schema,
    get_thread,
    mark_idle,
    mark_running,
    record_idle_from_child,
    sweep_expired,
    thread_unusable_reason,
)


class _FakeAgent:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.enabled_toolsets = ["terminal", "file"]
        self.disabled_toolsets = []
        self.model = "test-model"
        self.provider = "cursor"
        self.base_url = "https://example.com"
        self._delegate_depth = 1
        self._delegate_role = "leaf"
        self._delegate_output_schema = None


def test_record_idle_and_continue_eligibility(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ensure_schema()
    agent = _FakeAgent("child-sess-1")
    record = {
        "subagent_id": "sa-0-abc12345",
        "goal": "first task",
        "owner_agent_session_id": "parent-sess-1",
    }
    record_idle_from_child(record, agent)
    thread = get_thread(subagent_id="sa-0-abc12345")
    assert thread is not None
    assert thread["child_session_id"] == "child-sess-1"
    assert thread["status"] == "idle"
    assert thread_unusable_reason(thread) is None

    mark_running("sa-0-abc12345")
    thread = get_thread(subagent_id="sa-0-abc12345")
    assert thread["status"] == "running"
    assert "already running" in (thread_unusable_reason(thread) or "")

    mark_idle("sa-0-abc12345")
    thread = get_thread(subagent_id="sa-0-abc12345")
    assert thread["status"] == "idle"


def test_sweep_expired_marks_idle_threads(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ensure_schema()
    agent = _FakeAgent("child-sess-2")
    record_idle_from_child(
        {"subagent_id": "sa-0-deadbeef", "goal": "g", "owner_agent_session_id": "p"},
        agent,
    )
    # Force expiry in the past
    import sqlite3
    from hermes_constants import get_hermes_home

    db = get_hermes_home() / "state.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE delegation_threads SET idle_expires_at=? WHERE subagent_id=?",
            (time.time() - 10, "sa-0-deadbeef"),
        )
        conn.commit()
    assert sweep_expired() >= 1
    thread = get_thread(subagent_id="sa-0-deadbeef")
    assert thread["status"] == "expired"
