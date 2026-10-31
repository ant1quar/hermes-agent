"""Durable idle delegation threads: same child session_id for follow-up turns (delegate_task action=continue)."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger("tools.delegation_threads")

_LOCK = threading.Lock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS delegation_threads (
    subagent_id TEXT PRIMARY KEY,
    child_session_id TEXT NOT NULL,
    owner_agent_session_id TEXT,
    session_key TEXT,
    goal TEXT,
    status TEXT NOT NULL DEFAULT 'idle',
    toolsets_json TEXT,
    child_config_json TEXT,
    created_at REAL NOT NULL,
    last_activity_at REAL NOT NULL,
    idle_expires_at REAL,
    lifetime_expires_at REAL
);
CREATE INDEX IF NOT EXISTS idx_delegation_threads_child_session ON delegation_threads(child_session_id);
CREATE INDEX IF NOT EXISTS idx_delegation_threads_owner ON delegation_threads(owner_agent_session_id);
"""


def _db_path() -> Path:
    return Path(get_hermes_home()) / "state.db"




def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema() -> None:
    with _LOCK:
        conn = _connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()


def _cfg() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        return (load_config_readonly() or {}).get("orchestrator") or {}
    except Exception:
        return {}


def idle_ttl_seconds() -> float:
    minutes = float(_cfg().get("thread_idle_ttl_minutes") or 45)
    return max(60.0, minutes * 60.0)


def max_lifetime_seconds() -> float:
    hours = float(_cfg().get("thread_max_lifetime_hours") or 24)
    return max(3600.0, hours * 3600.0)


def record_idle_from_child(record: dict[str, Any], agent: Any, *, session_key: str = "") -> None:
    """Persist a finished subagent as an idle thread (rehydrate via action=continue)."""
    sid = str(record.get("subagent_id") or "")
    child_session_id = str(getattr(agent, "session_id", "") or "")
    if not sid or not child_session_id:
        return
    ensure_schema()
    now = time.time()
    toolsets = getattr(agent, "enabled_toolsets", None)
    disabled = getattr(agent, "disabled_toolsets", None)
    child_config = {
        "model": getattr(agent, "model", None),
        "provider": getattr(agent, "provider", None),
        "base_url": getattr(agent, "base_url", None),
        "delegate_depth": getattr(agent, "_delegate_depth", None),
        "delegate_role": getattr(agent, "_delegate_role", None),
        "output_schema": getattr(agent, "_delegate_output_schema", None),
        "delegate_route": getattr(agent, "_delegate_route", None),
        "disabled_toolsets": list(disabled) if isinstance(disabled, (list, tuple, set)) else None,
    }
    owner = str(record.get("owner_agent_session_id") or "")
    created = now
    with _LOCK, _connect() as conn:
        row = conn.execute(
            "SELECT created_at FROM delegation_threads WHERE subagent_id=?", (sid,)
        ).fetchone()
        if row and row["created_at"]:
            created = float(row["created_at"])
        conn.execute(
            """INSERT INTO delegation_threads
               (subagent_id, child_session_id, owner_agent_session_id, session_key, goal, status,
                toolsets_json, child_config_json, created_at, last_activity_at, idle_expires_at, lifetime_expires_at)
               VALUES (?, ?, ?, ?, ?, 'idle', ?, ?, ?, ?, ?, ?)
               ON CONFLICT(subagent_id) DO UPDATE SET
                 child_session_id=excluded.child_session_id,
                 owner_agent_session_id=excluded.owner_agent_session_id,
                 session_key=excluded.session_key,
                 goal=excluded.goal,
                 status='idle',
                 toolsets_json=excluded.toolsets_json,
                 child_config_json=excluded.child_config_json,
                 last_activity_at=excluded.last_activity_at,
                 idle_expires_at=excluded.idle_expires_at,
                 lifetime_expires_at=excluded.lifetime_expires_at""",
            (
                sid,
                child_session_id,
                owner,
                session_key or "",
                str(record.get("goal") or ""),
                json.dumps(list(toolsets) if toolsets else [], ensure_ascii=False),
                json.dumps(child_config, ensure_ascii=False, default=str),
                created,
                now,
                now + idle_ttl_seconds(),
                created + max_lifetime_seconds(),
            ),
        )
        conn.commit()


def mark_idle(subagent_id: str) -> None:
    ensure_schema()
    now = time.time()
    with _LOCK, _connect() as conn:
        conn.execute(
            """UPDATE delegation_threads SET status='idle', last_activity_at=?,
               idle_expires_at=? WHERE subagent_id=?""",
            (now, now + idle_ttl_seconds(), subagent_id),
        )
        conn.commit()


def mark_running(subagent_id: str) -> None:
    ensure_schema()
    now = time.time()
    with _LOCK, _connect() as conn:
        conn.execute(
            "UPDATE delegation_threads SET status='running', last_activity_at=? WHERE subagent_id=?",
            (now, subagent_id),
        )
        conn.commit()


def get_thread(*, subagent_id: str = "", child_session_id: str = "") -> Optional[dict[str, Any]]:
    ensure_schema()
    with _LOCK, _connect() as conn:
        if subagent_id:
            row = conn.execute(
                "SELECT * FROM delegation_threads WHERE subagent_id=?", (subagent_id,)
            ).fetchone()
        elif child_session_id:
            row = conn.execute(
                "SELECT * FROM delegation_threads WHERE child_session_id=? ORDER BY last_activity_at DESC LIMIT 1",
                (child_session_id,),
            ).fetchone()
        else:
            return None
    if row is None:
        return None
    return _row_to_dict(row)


def list_idle_for_owner(owner_agent_session_id: str, *, limit: int = 16) -> list[dict[str, Any]]:
    if not owner_agent_session_id:
        return []
    ensure_schema()
    sweep_expired()
    with _LOCK, _connect() as conn:
        rows = conn.execute(
            """SELECT * FROM delegation_threads
               WHERE owner_agent_session_id=? AND status='idle'
               ORDER BY last_activity_at DESC LIMIT ?""",
            (owner_agent_session_id, limit),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    cfg = {}
    try:
        cfg = json.loads(row["child_config_json"] or "{}")
    except json.JSONDecodeError:
        pass
    toolsets = []
    try:
        toolsets = json.loads(row["toolsets_json"] or "[]")
    except json.JSONDecodeError:
        pass
    return {
        "subagent_id": row["subagent_id"],
        "child_session_id": row["child_session_id"],
        "owner_agent_session_id": row["owner_agent_session_id"],
        "session_key": row["session_key"],
        "goal": row["goal"],
        "status": row["status"],
        "toolsets": toolsets,
        "child_config": cfg,
        "created_at": row["created_at"],
        "last_activity_at": row["last_activity_at"],
        "idle_expires_at": row["idle_expires_at"],
        "lifetime_expires_at": row["lifetime_expires_at"],
    }


def thread_unusable_reason(thread: dict[str, Any], now: Optional[float] = None) -> Optional[str]:
    """Return error text if continue is not allowed, else None."""
    now = now if now is not None else time.time()
    st = thread.get("status")
    if st == "expired":
        return "Thread expired; spawn a new worker."
    if st == "running":
        return "Thread is already running; use action=steer or wait for completion."
    idle_exp = float(thread.get("idle_expires_at") or 0)
    if idle_exp and now > idle_exp:
        return "Thread idle TTL expired; spawn a new worker with context from the last summary."
    life_exp = float(thread.get("lifetime_expires_at") or 0)
    if life_exp and now > life_exp:
        return "Thread max lifetime expired."
    return None


def sweep_expired() -> int:
    ensure_schema()
    now = time.time()
    with _LOCK, _connect() as conn:
        cur = conn.execute(
            """UPDATE delegation_threads SET status='expired'
               WHERE status IN ('idle', 'running')
                 AND ((idle_expires_at IS NOT NULL AND idle_expires_at < ?)
                      OR (lifetime_expires_at IS NOT NULL AND lifetime_expires_at < ?))""",
            (now, now),
        )
        conn.commit()
        return cur.rowcount or 0
