"""Global base layer for named profiles: the default (launch) home's ``config.yaml`` / ``.env`` /
``plugins/`` sit UNDER every named profile under ``profiles/<name>/``.

Precedence (low -> high): ``DEFAULT_CONFIG`` < base ``config.yaml`` < profile ``config.yaml`` <
managed overlay. Same for ``.env``: base < profile < managed. The default profile itself has no
base layer (it IS the base).

Never inherited: host-only gateway keys, per-user allowlists, and every messaging-channel
credential/section (bot tokens, ``telegram:`` blocks). A profile holding its own channel
credential is an independent adapter under multiplex, so an inherited bot token would make every
profile poll the host's bot; channels stay owned by the host. Likewise a profile without its own
``TELEGRAM_ALLOWED_USERS`` must not pick up another profile's users.
``HERMES_DISABLE_BASE_LAYER=1`` turns the layer off for the process.
"""

from __future__ import annotations

import copy
import fnmatch
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

from utils import fast_safe_load, file_signature

logger = logging.getLogger(__name__)

_HOST_ONLY_CONFIG_KEYS: tuple[str, ...] = (
    "_config_version",
    "gateway.multiplex_profiles",
    "gateway.auto_multiplex_migration",
    "gateway.profile_routes",
    "gateway.scale_to_zero",
    "gateway.restart_loop_guard",
    "gateway.api_server",
    "profiles.inherit_exclude",
)
_PROFILE_ONLY_ENV_PATTERNS: tuple[str, ...] = (
    "*_ALLOWED_USERS",
    "*_ALLOWED_CHATS",
    "*_ALLOWED_GROUPS",
    "HERMES_PROFILE",
    "HERMES_HOME",
)

# Built-in ``gateway.config.Platform`` ids (minus ``local``); not imported: gateway.config imports
# hermes_cli.config, which calls into this module while it is still initializing.
_CORE_PLATFORMS: frozenset[str] = frozenset((
    "telegram", "discord", "whatsapp", "whatsapp_cloud", "slack", "signal", "mattermost", "matrix",
    "email", "sms", "dingtalk", "api_server", "webhook", "msgraph_webhook", "feishu", "wecom",
    "wecom_callback", "weixin", "bluebubbles", "qqbot", "yuanbao", "relay", "irc",
))
# Used when no root is known.
_FALLBACK_CHANNEL_ENV_PREFIXES: tuple[str, ...] = (
    "TELEGRAM_", "DISCORD_", "SLACK_", "WHATSAPP_", "SIGNAL_", "MATTERMOST_", "MATRIX_",
    "FEISHU_", "DINGTALK_", "WECOM_", "BLUEBUBBLES_", "GOOGLE_CHAT_", "TEAMS_",
)

_NO_SIG = (0, 0, 0, 0)
_LOCK = threading.Lock()
# path -> (signature, parsed raw base config); a parse failure keeps serving the last good copy.
_CONFIG_CACHE: dict[str, tuple[tuple[int, ...], dict[str, Any]]] = {}
# path -> (raw object it was derived from, inheritable subset)
_STRIPPED_CACHE: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
# root path -> (platform ids, declared channel env names, channel env prefixes)
_CHANNEL_INDEX: dict[str, tuple[frozenset[str], frozenset[str], tuple[str, ...]]] = {}


def _disabled() -> bool:
    return os.environ.get("HERMES_DISABLE_BASE_LAYER", "").strip() == "1"


def base_home(home: Optional[Path] = None) -> Optional[Path]:
    """The default (launch) home when *home* (default: the active home) is a named profile under
    it, else None."""
    if _disabled():
        return None
    from hermes_constants import get_default_hermes_root, get_hermes_home

    try:
        active = Path(home).resolve() if home is not None else get_hermes_home().resolve()
        root = get_default_hermes_root().resolve()
    except OSError:
        return None
    if active == root or root not in active.parents:
        return None
    return root


def base_config_path(home: Optional[Path] = None) -> Optional[Path]:
    root = base_home(home)
    return root / "config.yaml" if root is not None else None


def base_config_signature(home: Optional[Path] = None) -> tuple[int, int, int, int]:
    """Change-detection key of the base ``config.yaml`` ((0, 0, 0, 0) when absent/disabled)."""
    path = base_config_path(home)
    if path is None:
        return _NO_SIG
    try:
        return file_signature(path.stat())
    except OSError:
        return _NO_SIG


_MISSING = object()


def _drop_dotted(config: dict[str, Any], dotted: str) -> None:
    """Remove *dotted*; parents emptied by the removal go too, sections empty in the file stay."""
    parts = dotted.split(".")
    chain: list[dict[str, Any]] = [config]
    for part in parts[:-1]:
        node = chain[-1].get(part)
        if not isinstance(node, dict):
            return
        chain.append(node)
    if chain[-1].pop(parts[-1], _MISSING) is _MISSING:
        return
    for parent, part in zip(reversed(chain[:-1]), reversed(parts[:-1])):
        if parent[part]:
            break
        parent.pop(part)


def _read_raw_base_config(path: Path) -> dict[str, Any]:
    key = str(path)
    try:
        sig = file_signature(path.stat())
    except OSError:
        _CONFIG_CACHE.pop(key, None)
        return {}
    with _LOCK:
        cached = _CONFIG_CACHE.get(key)
        if cached is not None and cached[0] == sig:
            return cached[1]
        try:
            with open(path, encoding="utf-8-sig") as f:
                raw = fast_safe_load(f) or {}
            if not isinstance(raw, dict):
                raise ValueError("base config.yaml is not a mapping")
        except Exception as exc:
            logger.warning("Global base config %s unreadable (%s); using last good copy", path, exc)
            return cached[1] if cached is not None else {}
        _CONFIG_CACHE[key] = (sig, raw)
        return raw


def _manifest_platforms(plugins_dir: Path) -> tuple[set[str], set[str]]:
    """``(platform ids, declared env names)`` from ``plugin.yaml`` files of platform plugins."""
    ids: set[str] = set()
    names: set[str] = set()
    if not plugins_dir.is_dir():
        return ids, names
    for manifest in plugins_dir.glob("*/plugin.yaml"):
        try:
            with open(manifest, encoding="utf-8-sig") as f:
                data = fast_safe_load(f) or {}
        except Exception:
            continue
        bundled_platform = manifest.parent.parent.name == "platforms"
        if not isinstance(data, dict) or not (bundled_platform or data.get("kind") == "platform"):
            continue
        ids.add(manifest.parent.name.replace("-", "_"))
        for section in ("requires_env", "optional_env"):
            for entry in data.get(section) or ():
                name = entry.get("name") if isinstance(entry, dict) else entry
                if isinstance(name, str) and name:
                    names.add(name)
    return ids, names


def _channel_index(root: Path) -> tuple[frozenset[str], frozenset[str], tuple[str, ...]]:
    """``(platform ids, declared channel env names, channel env prefixes)`` for *root*.

    Built from manifests on disk, never by importing plugins: this runs inside ``load_config``,
    which ``hermes_cli.config`` calls at import time, and plugin loader threads import that module.
    """
    key = str(root)
    with _LOCK:
        cached = _CHANNEL_INDEX.get(key)
    if cached is not None:
        return cached
    from hermes_cli import profile_channels as pc

    ids = set(_CORE_PLATFORMS)
    names: set[str] = set(pc._GATEWAY_POLICY_KEYS)
    for plugins_dir in (Path(__file__).resolve().parents[1] / "plugins" / "platforms", Path(root) / "plugins"):
        found_ids, found_names = _manifest_platforms(plugins_dir)
        ids |= found_ids
        names |= found_names
    # Prefixes shared with skills/tools (TWILIO_, EMAIL_) stay inheritable: the host owns no such adapter here.
    shared = tuple(p for prefixes in pc._SHARED_WITH_TOOLS.values() for p in prefixes)
    names = {n for n in names if not n.startswith(shared)}
    prefixes = tuple(sorted(
        {p for pid in ids for p in pc.platform_env_prefixes(pid)} - set(shared), key=len, reverse=True))
    index = (frozenset(ids), frozenset(names), prefixes)
    with _LOCK:
        _CHANNEL_INDEX[key] = index
    return index


def is_channel_env(name: str, root: Optional[Path] = None) -> bool:
    if root is None:
        return name.startswith(_FALLBACK_CHANNEL_ENV_PREFIXES)
    _, names, prefixes = _channel_index(root)
    return name in names or name.startswith(prefixes)


def _channel_config_paths(raw: dict[str, Any], root: Optional[Path]) -> list[str]:
    from hermes_cli.profile_channels import _channel_config_paths as paths_for

    platforms = _channel_index(root)[0] if root is not None else _CORE_PLATFORMS
    return [".".join(path) for path in paths_for(raw, platforms)]


def strip_host_only(raw: dict[str, Any], root: Optional[Path] = None) -> dict[str, Any]:
    """Copy of *raw* without host-only keys, messaging-channel blocks and the root's
    ``profiles.inherit_exclude`` paths."""
    config = copy.deepcopy(raw)
    extra = ((raw.get("profiles") or {}).get("inherit_exclude") or []) if isinstance(raw.get("profiles"), dict) else []
    for dotted in (*_HOST_ONLY_CONFIG_KEYS, *_channel_config_paths(raw, root), *(str(x) for x in extra if x)):
        _drop_dotted(config, dotted)
    if "max_turns" in config:
        agent = dict(config.get("agent") or {})
        agent.setdefault("max_turns", config["max_turns"])
        config["agent"] = agent
        config.pop("max_turns", None)
    return config


def load_base_config(home: Optional[Path] = None) -> dict[str, Any]:
    """Inheritable part of the base ``config.yaml`` (raw, unexpanded) for *home* (default: the
    active home), ``{}`` when it has no base layer."""
    path = base_config_path(home)
    if path is None:
        return {}
    raw = _read_raw_base_config(path)
    with _LOCK:
        memo = _STRIPPED_CACHE.get(str(path))
        if memo is not None and memo[0] is raw:
            return copy.deepcopy(memo[1])
    stripped = strip_host_only(raw, path.parent)
    with _LOCK:
        _STRIPPED_CACHE[str(path)] = (raw, stripped)
    return copy.deepcopy(stripped)


def read_layered_config_raw(config_path: Path) -> dict[str, Any]:
    """A profile's ``config.yaml`` as written (unexpanded, no defaults) over its base layer: for
    behavioral reads of ANOTHER profile's file. Never write the result back."""
    from hermes_cli.config import _deep_merge, read_user_config_raw
    raw = read_user_config_raw(config_path)
    base = load_base_config(Path(config_path).parent)
    return _deep_merge(base, raw) if base else raw


def is_profile_only_env(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in _PROFILE_ONLY_ENV_PATTERNS)


def base_env_path(home: Optional[Path] = None) -> Optional[Path]:
    root = base_home(home)
    return root / ".env" if root is not None else None


def load_base_env(home: Optional[Path] = None) -> dict[str, str]:
    """Inheritable assignments of the base ``.env`` for *home* (profile-only keys removed)."""
    path = base_env_path(home)
    if path is None or not path.is_file():
        return {}
    from agent.secret_scope import load_env_file

    try:
        values = load_env_file(path)
    except Exception as exc:
        logger.warning("Global base .env %s unreadable: %s", path, exc)
        return {}
    root = path.parent
    return {k: v for k, v in values.items() if not is_profile_only_env(k) and not is_channel_env(k, root)}


def base_plugins_dir(home: Optional[Path] = None) -> Optional[Path]:
    root = base_home(home)
    return root / "plugins" if root is not None else None


def invalidate_cache() -> None:
    with _LOCK:
        _CONFIG_CACHE.clear()
        _STRIPPED_CACHE.clear()
        _CHANNEL_INDEX.clear()
