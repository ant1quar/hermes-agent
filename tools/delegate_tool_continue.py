"""delegate_task(action=continue) — rehydrate a child session for another turn."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from tools.delegate_tool_registry import _owns_subagent_record, _resolve_session_lineage
from tools.registry import tool_error

logger = logging.getLogger("tools.delegate_tool")


def _owner_session_id(parent_agent: Any) -> str:
    return str(getattr(parent_agent, "session_id", "") or "")


def _owns_thread(thread: dict[str, Any], parent_agent: Any) -> bool:
    owner = str(thread.get("owner_agent_session_id") or "")
    parent_sid = _owner_session_id(parent_agent)
    if not owner or not parent_sid:
        return False
    if owner == parent_sid:
        return True
    return _resolve_session_lineage(owner, parent_agent) in {
        parent_sid,
        _resolve_session_lineage(parent_sid, parent_agent),
    }


def run_continue_delegation(
    parent_agent: Any,
    *,
    message: Optional[str],
    subagent_id: Optional[str] = None,
    child_session_id: Optional[str] = None,
) -> str:
    text = (message or "").strip()
    if not text:
        return tool_error("action='continue' requires a non-empty 'message' (the follow-up user task).")
    if not subagent_id and not child_session_id:
        return tool_error("action='continue' requires subagent_id or child_session_id from a prior worker.")

    from tools.delegation_threads import get_thread, mark_running, sweep_expired, thread_unusable_reason

    sweep_expired()
    thread = get_thread(subagent_id=subagent_id or "", child_session_id=child_session_id or "")
    if thread is None:
        return tool_error(
            "No idle delegation thread found for that id. The worker may have expired or never "
            "registered — use action='spawn' for a new task."
        )
    if not _owns_thread(thread, parent_agent):
        return tool_error("That delegation thread belongs to another conversation.")

    reason = thread_unusable_reason(thread)
    if reason:
        return tool_error(reason)

    sid = thread["subagent_id"]
    mark_running(sid)
    try:
        return _dispatch_continue(parent_agent, thread, sid, text)
    except Exception as exc:
        from tools.delegation_threads import mark_idle

        mark_idle(sid)
        logger.exception("continue dispatch failed for %s", sid)
        return tool_error(f"Could not continue worker {sid}: {exc}")


def _dispatch_continue(parent_agent: Any, thread: dict[str, Any], sid: str, text: str) -> str:
    from tools.delegate_tool import DEFAULT_MAX_ITERATIONS, _get_max_concurrent_children, _load_config
    from tools.delegate_tool_dispatch import _Batch, _capture_origin, _run_batch
    from tools.delegate_tool_results import _build_child_preserving_parent_tools
    from tools.delegation_live_log import create_live_transcripts

    cfg = _load_config()
    default_max_iter = cfg.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    child_cfg = thread.get("child_config") or {}
    toolsets = thread.get("toolsets") or None
    disabled = child_cfg.get("disabled_toolsets")
    schema = child_cfg.get("output_schema")
    if isinstance(schema, dict):
        from tools.delegation_output_schema import append_output_contract

        # Continued children get no system prompt, so the contract must ride on the turn itself.
        text = append_output_contract(text, schema)

    child = _build_child_preserving_parent_tools(
        task_index=0,
        goal=text,
        context=None,
        toolsets=toolsets,
        model=child_cfg.get("model"),
        max_iterations=default_max_iter,
        task_count=1,
        parent_agent=parent_agent,
        existing_child_session_id=thread["child_session_id"],
        fixed_subagent_id=sid,
        continue_disabled_toolsets=disabled,
    )

    if isinstance(schema, dict):
        child._delegate_output_schema = schema

    live_home = None
    with __import__("contextlib").suppress(Exception):
        from tools.delegate_tool import _parent_live_home

        live_home = _parent_live_home(parent_agent)

    task = {"goal": text, "context": None}
    live_deleg_id, live_writers, live_paths = create_live_transcripts(
        [task],
        None,
        model=child_cfg.get("model"),
        provider=child_cfg.get("provider"),
        home=live_home,
    )
    if live_writers:
        from tools.delegation_live_log import wrap_progress_callback

        child.tool_progress_callback = wrap_progress_callback(
            getattr(child, "tool_progress_callback", None), live_writers[0]
        )
        child._live_transcript_path = str(live_writers[0].path)

    child._delegation_id = live_deleg_id
    origin = _capture_origin()
    creds = {
        "model": child_cfg.get("model") or getattr(child, "model", None),
        "provider": child_cfg.get("provider") or getattr(child, "provider", None),
    }
    batch = _Batch(
        [task],
        [(0, task, child)],
        parent_agent,
        creds,
        None,
        child_cfg.get("delegate_role") or "leaf",
        _get_max_concurrent_children(),
        live_deleg_id,
        live_writers,
        live_paths,
        *origin,
        time.monotonic(),
        live_home=live_home,
    )
    # Continue always uses background delivery when the session supports it (same as spawn on gateway).
    return _run_batch(batch, background=True)
