"""T1-24: structured-output schema on delegate_task.

Per-task ``output_schema`` (JSON Schema object): the child receives the
schema as an explicit output contract, the parent validates the child's
final answer with jsonschema, and on failure sends exactly ONE bounded
retry turn carrying the validation errors. Result entries gain
``schema_valid`` / ``schema_errors`` / ``schema_retries`` ONLY when a
schema was requested — schema-less calls keep a byte-identical result
shape (wire-shape pinning).

Pattern from: github/copilot-cli ctx.agent(prompt, {schema}) — PATTERN
ONLY, zero code/prompt text copied (proprietary).
"""

import json
import threading
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _run_single_child,
    delegate_task,
)
from tools.delegation_output_schema import (
    append_output_contract,
    build_retry_message,
    extract_json_candidate,
    validate_output,
)

ADDRESS_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "zip": {"type": "string"},
    },
    "required": ["city"],
}


# ---------------------------------------------------------------------------
# Helper-module unit tests
# ---------------------------------------------------------------------------


class TestValidateOutput:
    def test_valid_json_matching_schema(self):
        ok, errors = validate_output('{"city": "Berlin"}', ADDRESS_SCHEMA)
        assert ok is True
        assert errors == []

    def test_json_violating_schema_reports_errors(self):
        ok, errors = validate_output('{"zip": "10115"}', ADDRESS_SCHEMA)
        assert ok is False
        assert errors
        assert any("city" in e for e in errors)

    def test_non_json_text_reports_parse_error(self):
        ok, errors = validate_output("I could not produce JSON, sorry.", ADDRESS_SCHEMA)
        assert ok is False
        assert errors

    def test_code_fenced_json_is_accepted(self):
        text = '```json\n{"city": "Oslo"}\n```'
        ok, errors = validate_output(text, ADDRESS_SCHEMA)
        assert ok is True
        assert errors == []

    def test_json_embedded_in_prose_is_extracted(self):
        text = 'Here is the result:\n{"city": "Lima"}\nHope that helps!'
        ok, _ = validate_output(text, ADDRESS_SCHEMA)
        assert ok is True

    def test_empty_text_is_invalid(self):
        ok, errors = validate_output("", ADDRESS_SCHEMA)
        assert ok is False
        assert errors

    def test_two_streamed_attempts_validate_on_the_last_object(self):
        """A provider retry that streamed both attempts: 'Extra data' used to reject the whole answer."""
        text = ('Investigating X.\n\n\n{"city": "Old", "zip": "1"}'
                'Investigating X.\n\n\n{"city": "New", "zip": "2"}')
        ok, errors = validate_output(text, ADDRESS_SCHEMA)
        assert ok is True and errors == []
        assert json.loads(extract_json_candidate(text, ADDRESS_SCHEMA))["city"] == "New"

    def test_prose_braces_before_object_are_skipped(self):
        ok, _ = validate_output('Paths use {HERMES_HOME} and [1].\n{"city": "Rome"}', ADDRESS_SCHEMA)
        assert ok is True

    def test_inner_array_is_not_taken_for_the_answer(self):
        text = 'Done.\n```json\n{"city": "Oslo", "plan": {"parts": ["a", "b"]}}\n```'
        ok, _ = validate_output(text, ADDRESS_SCHEMA)
        assert ok is True

    def test_last_object_errors_are_reported(self):
        ok, errors = validate_output('{"city": "A"} then {"zip": "1"}', {**ADDRESS_SCHEMA, "required": ["zip", "x"]})
        assert ok is False
        assert any("'x'" in e for e in errors)




class TestPromptPlumbing:
    def test_contract_block_carries_schema(self):
        out = append_output_contract("base context", ADDRESS_SCHEMA)
        assert "base context" in out
        assert "OUTPUT CONTRACT" in out
        assert '"city"' in out


    def test_retry_message_carries_verbatim_errors(self):
        msg = build_retry_message(["'city' is a required property"])
        assert "'city' is a required property" in msg
        assert "JSON" in msg

    def test_retry_message_names_required_keys(self):
        msg = build_retry_message(["x"], ADDRESS_SCHEMA)
        assert "Required top-level keys: city." in msg

    def test_prose_only_retry_asks_to_finish_the_work(self):
        msg = build_retry_message(["Response is not valid JSON"], ADDRESS_SCHEMA, no_json=True)
        assert "continue it with tool calls" in msg
        assert "rejected by the output contract" not in msg


# ---------------------------------------------------------------------------
# Tool-schema surface (one-time static field)
# ---------------------------------------------------------------------------


class TestToolSchemaSurface:

    def test_output_schema_advertised_per_task_only(self):
        """output_schema is advertised inside tasks[] items (the only spawn
        shape); the legacy top-level param stays handler-accepted but out
        of the schema."""
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        assert "output_schema" not in props
        task_props = props["tasks"]["items"]["properties"]
        assert task_props["output_schema"]["type"] == "object"


# ---------------------------------------------------------------------------
# _run_single_child validation + bounded retry
# ---------------------------------------------------------------------------


class _StubChild:
    """Minimal child agent double (mirrors test_delegate_kanban_isolation)."""

    tool_progress_callback = None
    _delegate_saved_tool_names: list = []
    _credential_pool = None
    _subagent_id = None  # skip registry
    _delegate_depth = 1
    _parent_subagent_id = None
    _delegate_output_schema: dict | None = None
    model = "test-model"
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_estimated_cost_usd = 0.0
    session_reasoning_tokens = 0

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list = []

    def get_activity_summary(self):
        return {"api_call_count": 1, "max_iterations": 5, "current_tool": None}

    def run_conversation(self, user_message, task_id=None, **_kwargs):
        self.calls.append(user_message)
        text = self.responses.pop(0)
        return {
            "final_response": text,
            "completed": True,
            "api_calls": 1,
            "messages": [],
        }

    def close(self):
        return None


class _StubParent:
    _current_task_id = None
    _delegate_depth = 0

    def _touch_activity(self, _desc):
        return None


def _run(child):
    return _run_single_child(0, "produce the address", child, _StubParent())


class TestRunSingleChildSchemaValidation:
    def test_valid_first_try_no_retry(self):
        child = _StubChild(['{"city": "Berlin"}'])
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["status"] == "completed"
        assert entry["schema_valid"] is True
        assert "schema_errors" not in entry
        assert len(child.calls) == 1

    def test_invalid_then_retry_then_valid(self):
        child = _StubChild(["not json at all", '{"city": "Oslo"}'])
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["schema_valid"] is True
        assert entry["schema_retries"] == 1
        # retry turn carried the validation errors
        assert len(child.calls) == 2
        assert "rejected" in child.calls[1] or "JSON" in child.calls[1]
        # final summary is the retried (valid) answer
        assert json.loads(entry["summary"])["city"] == "Oslo"

    def test_invalid_twice_surfaces_errors_and_stops(self):
        child = _StubChild(["nope", "still nope"])
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["schema_valid"] is False
        assert entry["schema_errors"]
        assert entry["schema_retries"] == 1
        # exactly ONE retry — bounded
        assert len(child.calls) == 2

    def test_retry_exception_degrades_to_invalid(self):
        child = _StubChild(["nope"])
        child._delegate_output_schema = ADDRESS_SCHEMA

        original = child.run_conversation

        def flaky(user_message, task_id=None, **kw):
            if child.calls:
                raise RuntimeError("child died on retry")
            return original(user_message, task_id=task_id, **kw)

        child.run_conversation = flaky
        entry = _run(child)
        assert entry["schema_valid"] is False
        assert entry["schema_errors"]

    def test_no_schema_keeps_legacy_result_shape(self):
        """Schema-less calls must not gain new keys (wire-shape pinning)."""
        child = _StubChild(['{"city": "Berlin"}'])
        entry = _run(child)
        assert "schema_valid" not in entry
        assert "schema_errors" not in entry
        assert "schema_retries" not in entry
        assert len(child.calls) == 1

    def test_failed_child_skips_validation(self):
        """A child with no output never gets a schema retry turn."""
        child = _StubChild([""])
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["status"] == "failed"
        assert len(child.calls) == 1
        assert entry.get("schema_valid") is False

    def test_schema_failure_returns_raw_text_with_schema_valid_false(self):
        """A final answer that still violates the contract after the bounded retry is NOT discarded: the
        child did the work (audits of 400-4100 s were written off over a stray fence or one missing
        field). The parent gets the raw text as ``summary`` with status "completed", ``schema_valid``
        false, ``schema_errors`` populated and a ``schema_note`` saying the text is unvalidated."""
        prose = 'Findings:\n```json\n{"town": "Oslo"}\n```\nlet me know.'
        child = _StubChild([prose, prose])
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["status"] == "completed"
        assert entry["summary"] == prose
        assert entry["schema_valid"] is False
        assert entry["schema_errors"] and entry["schema_retries"] == 1
        assert "UNVALIDATED" in entry["schema_note"]
        assert "error" not in entry
        assert len(child.calls) == 2  # exactly one bounded retry

    def test_prose_wrapped_array_answer_validates(self):
        """A fenced JSON array with prose around it is the answer, not a violation: the extractor
        used to slice it to its first..last object and reject every array-shaped result."""
        text = 'Verdicts below.\n```json\n[{"n": 1}, {"n": 2}]\n```\n'
        child = _StubChild([text])
        child._delegate_output_schema = {"type": "array", "items": {"type": "object"}}
        entry = _run(child)
        assert entry["schema_valid"] is True and len(child.calls) == 1


    def test_retry_turn_continues_the_child_conversation(self):
        """The retry turn without history saw only the rejection, could not recover the task and answered
        with a schema-valid 'resend the task' that the parent relayed to the user."""
        from tools.delegate_tool_child_run import _validate_child_output_schema

        history = [{"role": "user", "content": "produce the address"},
                   {"role": "assistant", "content": "Checking the registry."}]
        seen: dict = {}

        class _Child(_StubChild):
            def run_conversation(self, user_message, task_id=None, conversation_history=None, **_kwargs):
                seen["history"] = conversation_history
                seen["message"] = user_message
                return {"final_response": '{"city": "Oslo"}', "api_calls": 1,
                        "messages": list(conversation_history or []) + [
                            {"role": "user", "content": user_message},
                            {"role": "assistant", "content": '{"city": "Oslo"}'}]}

        child = _Child([])
        child._delegate_output_schema = ADDRESS_SCHEMA
        result = {"final_response": "Checking the registry.", "api_calls": 1, "messages": history}
        outcome = _validate_child_output_schema(child, result, 0, "child-0", None)
        assert outcome.valid is True and outcome.retries == 1 and outcome.first_errors
        assert seen["history"] == history
        assert "continue it with tool calls" in seen["message"]
        assert len(result["messages"]) == 4  # history once, then the retry pair

    def test_valid_glued_answer_delivers_only_the_accepted_object(self):
        from tools.delegate_tool_child_run import _validate_child_output_schema

        child = _StubChild([])
        child._delegate_output_schema = ADDRESS_SCHEMA
        result = {"final_response": 'A.\n{"city": "Old"}B.\n{"city": "New"}', "api_calls": 1, "messages": []}
        outcome = _validate_child_output_schema(child, result, 0, "child-0", None)
        assert outcome.valid is True and outcome.retries == 0 and not child.calls
        assert json.loads(result["final_response"]) == {"city": "New"}

    def test_retry_turn_runs_in_delegated_child_context(self, monkeypatch):
        """The retry is a second run_conversation on the child, issued from the parent's
        thread where HERMES_KANBAN_TASK is set. Unwrapped it carries the worker's identity,
        so the kanban stop guard nudges the child toward a board tool it does not have
        (#109735 / #87671)."""
        from agent.delegation_context import is_delegated_child_context
        from tools.delegate_tool_child_run import _validate_child_output_schema

        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
        seen: list = []

        class _Child(_StubChild):
            session_id = "sess-child"

            def run_conversation(self, user_message, task_id=None, **_kwargs):
                seen.append(is_delegated_child_context())
                return super().run_conversation(user_message, task_id, **_kwargs)

        child = _Child(['{"city": "Berlin"}'])
        child._delegate_output_schema = ADDRESS_SCHEMA
        _validate_child_output_schema(child, {"final_response": "not json", "api_calls": 1}, 0, "child-0", None)
        assert seen == [True]
        assert is_delegated_child_context() is False


class TestWakeSchemaRetryMarker:
    def test_header_and_note_name_the_correction_turn(self):
        import re
        from tools.process_registry_notifications import _format_batch_delegation

        goal = 'Merge duplicates.\n\nOUTPUT CONTRACT (machine-validated):\n{\n  "type": "object"\n}'
        evt = {"goals": [goal], "results": [{
            "task_index": 0, "status": "completed", "summary": '{"ok": false}', "api_calls": 5,
            "duration_seconds": 86.28, "schema_valid": True, "schema_retries": 1,
            "schema_first_errors": ["Response is not valid JSON: Extra data: line 1 column 608 (char 607)"],
        }]}
        text = _format_batch_delegation(evt, "deleg_0d0d0393", 0.0)
        assert "schema_retries=1) ---" in text
        assert "[OUTPUT CONTRACT — the subagent's first final answer was rejected (Response is not valid" in text
        # orchestrator-fsm keys the redirect on this exact header shape (multi-line goal included)
        assert re.search(r"\(status=[^)\n]*\bschema_retries=[1-9][^)\n]*\) ---", text)

    def test_no_marker_without_retry(self):
        from tools.process_registry_notifications import _format_batch_delegation

        evt = {"goals": ["g"], "results": [{"task_index": 0, "status": "completed", "summary": "{}"}]}
        assert "schema_retries" not in _format_batch_delegation(evt, "deleg_1", 0.0)


class TestWakeSchemaRetryMarker:
    def test_header_and_note_name_the_correction_turn(self):
        import re
        from tools.process_registry_notifications import _format_batch_delegation

        goal = 'Merge duplicates.\n\nOUTPUT CONTRACT (machine-validated):\n{\n  "type": "object"\n}'
        evt = {"goals": [goal], "results": [{
            "task_index": 0, "status": "completed", "summary": '{"ok": false}', "api_calls": 5,
            "duration_seconds": 86.28, "schema_valid": True, "schema_retries": 1,
            "schema_first_errors": ["Response is not valid JSON: Extra data: line 1 column 608 (char 607)"],
        }]}
        text = _format_batch_delegation(evt, "deleg_0d0d0393", 0.0)
        assert "schema_retries=1) ---" in text
        assert "[OUTPUT CONTRACT — the subagent's first final answer was rejected (Response is not valid" in text
        # orchestrator-fsm keys the redirect on this exact header shape (multi-line goal included)
        assert re.search(r"\(status=[^)\n]*\bschema_retries=[1-9][^)\n]*\) ---", text)

    def test_no_marker_without_retry(self):
        from tools.process_registry_notifications import _format_batch_delegation

        evt = {"goals": ["g"], "results": [{"task_index": 0, "status": "completed", "summary": "{}"}]}
        assert "schema_retries" not in _format_batch_delegation(evt, "deleg_1", 0.0)


# ---------------------------------------------------------------------------
# delegate_task dispatch-time schema handling
# ---------------------------------------------------------------------------


def _make_mock_parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    return parent


class TestDelegateTaskDispatch:
    def test_non_dict_output_schema_rejected(self):
        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch(
                "tools.delegate_tool._resolve_delegation_credentials",
                return_value={
                    "provider": None,
                    "model": None,
                    "base_url": None,
                    "api_key": None,
                    "api_mode": None,
                },
            ),
        ):
            out = delegate_task(
                tasks=[
                    {"goal": "Summarize the release notes for module A", "output_schema": "not-a-dict"},
                    {"goal": "Summarize the release notes for module B"},
                ],
                parent_agent=_make_mock_parent(),
            )
        payload = json.loads(out)
        assert payload.get("error")
        assert "output_schema" in payload["error"]

    def test_invalid_json_schema_rejected_at_dispatch(self):
        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch(
                "tools.delegate_tool._resolve_delegation_credentials",
                return_value={
                    "provider": None,
                    "model": None,
                    "base_url": None,
                    "api_key": None,
                    "api_mode": None,
                },
            ),
        ):
            out = delegate_task(
                tasks=[
                    {"goal": "Summarize the release notes for module A", "output_schema": {"type": 42}},
                    {"goal": "Summarize the release notes for module B"},
                ],
                parent_agent=_make_mock_parent(),
            )
        payload = json.loads(out)
        assert payload.get("error")
        assert "output_schema" in payload["error"]

    def test_child_receives_contract_and_schema_attr(self):
        """The built child carries the schema attr and its context gains
        the output-contract block."""
        captured = {}

        def fake_build(**kwargs):
            captured.update(kwargs)
            child = _StubChild(['{"city": "Rio"}'])
            return child

        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch(
                "tools.delegate_tool._resolve_delegation_credentials",
                return_value={
                    "provider": None,
                    "model": None,
                    "base_url": None,
                    "api_key": None,
                    "api_mode": None,
                },
            ),
            patch(
                "tools.delegate_tool._build_child_preserving_parent_tools",
                side_effect=fake_build,
            ),
        ):
            out = delegate_task(
                goal="produce the address",
                context="base context",
                output_schema=ADDRESS_SCHEMA,
                parent_agent=_make_mock_parent(),
            )
        payload = json.loads(out)
        assert "OUTPUT CONTRACT" in (captured.get("context") or "")
        results = payload.get("results") or []
        assert results and results[0].get("schema_valid") is True
