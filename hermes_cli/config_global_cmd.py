"""``hermes config ... --global`` (write the shared base layer) and ``config show --origin``."""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from hermes_cli.config_base_layer import _HOST_ONLY_CONFIG_KEYS, base_home

_SECRET_KEY_RE = re.compile(r"(api_?key|token|secret|password)", re.IGNORECASE)


@contextmanager
def global_home() -> Iterator[None]:
    """Run config reads/writes against the default (base) home instead of the active profile."""
    from hermes_constants import get_default_hermes_root, reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(get_default_hermes_root())
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _is_host_only(key: str) -> bool:
    return any(key == k or key.startswith(k + ".") for k in _HOST_ONLY_CONFIG_KEYS)


def run_global(fn: Callable[..., Any], key: str, *args: Any) -> Any:
    if _is_host_only(key):
        print(f"Note: {key} is a gateway-host setting; named profiles do not inherit it.")
    with global_home():
        return fn(key, *args)


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(node, dict) and node:
        out: dict[str, Any] = {}
        for k, v in node.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: node} if prefix else {}


def _display(key: str, value: Any) -> str:
    if value and _SECRET_KEY_RE.search(key.rsplit(".", 1)[-1]):
        return "***"
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def config_origins() -> dict[str, tuple[Any, str]]:
    """``{dotted_key: (effective_value, origin)}`` for every leaf NOT coming from defaults."""
    from hermes_cli import managed_scope
    from hermes_cli.config import load_config_readonly, read_raw_config_readonly
    from hermes_cli.config_base_layer import load_base_config

    effective = _flatten(load_config_readonly())
    profile_keys = set(_flatten(read_raw_config_readonly()))
    base_keys = set(_flatten(load_base_config())) if base_home() is not None else set()
    managed_keys = set(managed_scope.managed_config_keys() or ())

    def _origin(key: str) -> str:
        def _hit(keys: set[str]) -> bool:
            return key in keys or any(key.startswith(k + ".") for k in keys)

        if _hit(managed_keys):
            return "managed"
        if _hit(profile_keys):
            return "profile" if base_home() is not None else "config"
        if _hit(base_keys):
            return "global"
        return "default"

    out = {}
    for key, value in sorted(effective.items()):
        origin = _origin(key)
        if origin != "default":
            out[key] = (value, origin)
    return out


def show_config_origins() -> None:
    from hermes_cli.config import get_config_path

    root = base_home()
    print(f"Profile config: {get_config_path()}")
    print(f"Global config:  {root / 'config.yaml' if root is not None else '(none: active home is the base)'}")
    print()
    for key, (value, origin) in config_origins().items():
        print(f"  {'[' + origin + ']':10} {key} = {_display(key, value)}")
