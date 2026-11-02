"""Structured-output schema helpers for delegate_task.

Optional per-task ``output_schema`` (a JSON Schema object): the child gets an
OUTPUT CONTRACT block appended to its context, the parent validates the final
answer with jsonschema, and on failure sends exactly ONE bounded retry turn
carrying the validation errors verbatim (more retries make frontier models
drop fields that were right the first time; the schema is never re-pasted, so
the retry turn must carry the child's conversation history).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def coerce_output_schema(raw: Any) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """``(schema, None)`` when usable, ``(None, error)`` when not; ``None`` input
    passes through as ``(None, None)`` (no schema requested)."""
    if raw is None:
        return None, None
    if isinstance(raw, str):
        # Models sometimes double-encode the schema as a JSON string.
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None, "output_schema must be a JSON Schema object, got a non-JSON string."
        if not isinstance(raw, dict):
            return None, "output_schema must be a JSON Schema object."
    if not isinstance(raw, dict):
        return None, f"output_schema must be a JSON Schema object, got {type(raw).__name__}."
    try:
        from jsonschema.validators import validator_for  # type: ignore[import-untyped]
        validator_for(raw).check_schema(raw)
    except ImportError:
        # Degrade to accepting the dict as-is so delegation still works without jsonschema.
        logger.debug("jsonschema unavailable; skipping output_schema meta-validation")
    except Exception as exc:
        return None, f"output_schema is not a valid JSON Schema: {exc}"
    return raw, None


def append_output_contract(context: Optional[str], schema: dict[str, Any]) -> str:
    """Append the explicit output contract block to a child's context."""
    try:
        schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        schema_text = str(schema)
    block = ("OUTPUT CONTRACT (machine-validated):\n"
             "Your FINAL response must be ONLY the JSON value that validates against this JSON "
             "Schema — no prose before or after it, no code fence, no explanation. Anything else "
             "costs a correction turn and, if it fails again, is handed to the caller unvalidated.\n"
             f"{schema_text}")
    base = (context or "").rstrip()
    return f"{base}\n\n{block}" if base else block


def _strip_fence(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[: -3]
        raw = raw.strip()
        if raw.lower().startswith("json\n"):
            raw = raw.split("\n", 1)[1]
    return raw


def json_values(text: str) -> list[tuple[str, Any]]:
    """Every top-level JSON object/array embedded in ``text`` as ``(source, value)``, in order.

    Scans each ``{``/``[`` with ``raw_decode`` and resumes after a decoded value, so nested values are not
    listed twice, prose braces (``{HERMES_HOME}``) are skipped, and back-to-back answers (a provider that
    streamed two attempts, or a reply followed by a second block) come out as separate values."""
    raw = _strip_fence(text)
    decoder = json.JSONDecoder()
    found: list[tuple[str, Any]] = []
    pos = 0
    while True:
        starts = [i for i in (raw.find("{", pos), raw.find("[", pos)) if i >= 0]
        if not starts:
            return found
        start = min(starts)
        try:
            value, end = decoder.raw_decode(raw, start)
        except ValueError:
            pos = start + 1
            continue
        found.append((raw[start:end], value))
        pos = end


def extract_json_candidate(text: str, schema: Optional[dict[str, Any]] = None) -> str:
    """The JSON source the validator judges: the last embedded value that satisfies ``schema`` (any schema-less
    or no-match call takes the last embedded value), else the fence-stripped text itself."""
    values = json_values(text)
    if not values:
        return _strip_fence(text)
    if schema is not None:
        for source, value in reversed(values):
            if not _schema_errors(value, schema):
                return source
    return values[-1][0]


def _schema_errors(value: Any, schema: dict[str, Any]) -> list[str]:
    try:
        from jsonschema.validators import validator_for  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("jsonschema unavailable; accepting parsed JSON without validation")
        return []
    validator = validator_for(schema)(schema)
    errors = sorted(validator.iter_errors(value), key=lambda e: list(e.absolute_path))
    return [  # bound error volume for the retry prompt
        "$" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in err.absolute_path) + f": {err.message}"
        for err in errors[:10]]


def validate_output(text: str, schema: dict[str, Any]) -> tuple[bool, list[str]]:
    """``(True, [])`` or ``(False, errors)`` with strings suitable for the retry turn."""
    if not _strip_fence(text or "").strip():
        return False, ["Response was empty — expected a JSON object matching the schema."]
    values = json_values(text or "")
    if not values:
        try:
            scalar = json.loads(_strip_fence(text or ""))
        except (ValueError, TypeError) as exc:
            return False, [f"Response is not valid JSON: {exc}"]
        errors = _schema_errors(scalar, schema)
        return not errors, errors
    for _source, value in reversed(values):
        if not _schema_errors(value, schema):
            return True, []
    return False, _schema_errors(values[-1][1], schema)


def has_json_value(text: str) -> bool:
    return bool(json_values(text or ""))


def build_retry_message(
    errors: list[str], schema: Optional[dict[str, Any]] = None, *, no_json: bool = False,
) -> str:
    """Single bounded retry turn: errors verbatim plus the required keys; the full schema is NOT re-pasted.

    ``no_json``: the turn ended with prose only (usually an action announcement with no tool call), so the
    child is told to finish the work first instead of re-wrapping an answer it never produced."""
    error_block = "\n".join(f"- {e}" for e in errors)
    required = schema.get("required") if isinstance(schema, dict) else None
    keys = (f" Required top-level keys: {', '.join(str(k) for k in required)}."
            if isinstance(required, list) and required else "")
    if no_json:
        return ("Your previous turn ended without the final JSON answer required by the output contract. "
                "Validation errors:\n" f"{error_block}\n\n"
                "If the task is not finished, continue it with tool calls now. Then end with ONLY the JSON "
                "object matching the OUTPUT CONTRACT schema from your task context — no prose before or after "
                f"it, no code fence.{keys}")
    return ("Your previous final response was rejected by the output contract "
            "validator. Validation errors:\n" f"{error_block}\n\n"
            "Reply with ONLY the corrected JSON object matching the OUTPUT "
            f"CONTRACT schema from your task context. No prose, no explanations.{keys}")
