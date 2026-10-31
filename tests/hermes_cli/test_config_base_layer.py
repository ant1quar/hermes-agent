"""Global base layer: the root home's config.yaml / .env / plugins sit under every named profile."""

import os
from pathlib import Path

import pytest

from hermes_cli import config as config_mod
from hermes_cli import config_base_layer as base_layer


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def layout(tmp_path, monkeypatch):
    root = tmp_path / "root"
    profile = root / "profiles" / "ivan"
    _write(root / "config.yaml", (
        "_config_version: 51\n"
        "agent:\n  reasoning_effort: low\n  max_turns: 77\n"
        "display:\n  skin: global-skin\n"
        "gateway:\n  multiplex_profiles: false\n"
        "  profile_routes:\n    - {platform: telegram, user_id: '1', profile: ivan}\n"
    ))
    _write(profile / "config.yaml", "_config_version: 51\nagent:\n  reasoning_effort: high\n")
    _write(root / ".env", (
        "CURSOR_API_KEY=global-key\nTELEGRAM_ALLOWED_USERS=1,2\nTELEGRAM_BOT_TOKEN=123:abc\n"
        "DISCORD_BOT_TOKEN=d\nSHARED_TOKEN=g\n"
    ))
    _write(profile / ".env", "SHARED_TOKEN=p\n")
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.delenv("HERMES_DISABLE_BASE_LAYER", raising=False)
    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()
    base_layer.invalidate_cache()
    yield root, profile
    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()
    base_layer.invalidate_cache()


def test_base_home_only_for_named_profiles(layout, monkeypatch):
    root, _profile = layout
    assert base_layer.base_home() == root.resolve()
    monkeypatch.setenv("HERMES_HOME", str(root))
    assert base_layer.base_home() is None


def test_precedence_default_global_profile(layout):
    cfg = config_mod.load_config()
    assert cfg["agent"]["reasoning_effort"] == "high"  # profile wins
    assert cfg["agent"]["max_turns"] == 77  # inherited from global
    assert cfg["display"]["skin"] == "global-skin"


def test_host_only_keys_not_inherited(layout):
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    cfg = config_mod.load_config()
    assert cfg["gateway"]["multiplex_profiles"] == DEFAULT_CONFIG["gateway"]["multiplex_profiles"]
    assert not cfg.get("gateway", {}).get("profile_routes")


def test_inherit_exclude_extends_host_only(layout):
    root, _ = layout
    text = (root / "config.yaml").read_text() + "profiles:\n  inherit_exclude: [display.skin]\n"
    _write(root / "config.yaml", text)
    assert "skin" not in (base_layer.load_base_config().get("display") or {})


def test_global_edit_invalidates_profile_cache(layout):
    root, _ = layout
    assert config_mod.load_config()["display"]["skin"] == "global-skin"
    text = (root / "config.yaml").read_text().replace("global-skin", "changed-skin")
    _write(root / "config.yaml", text)
    os.utime(root / "config.yaml", ns=(1, 10**18))
    assert config_mod.load_config()["display"]["skin"] == "changed-skin"


def test_save_in_profile_does_not_copy_global_values(layout):
    _, profile = layout
    cfg = config_mod.load_config()
    cfg["agent"]["reasoning_effort"] = "medium"
    config_mod.save_config(cfg)
    raw = (profile / "config.yaml").read_text()
    assert "medium" in raw
    assert "global-skin" not in raw
    assert "max_turns: 77" not in raw


def test_broken_global_yaml_keeps_last_good(layout):
    root, _ = layout
    assert base_layer.load_base_config()["display"]["skin"] == "global-skin"
    _write(root / "config.yaml", "agent: [unclosed\n")
    os.utime(root / "config.yaml", ns=(1, 10**18))
    assert base_layer.load_base_config()["display"]["skin"] == "global-skin"


def test_base_env_drops_allowlists_and_profile_wins(layout):
    _ = layout
    env = config_mod.load_env()
    assert env["CURSOR_API_KEY"] == "global-key"
    assert env["SHARED_TOKEN"] == "p"
    assert "TELEGRAM_ALLOWED_USERS" not in env


def test_effective_and_layered_reads_of_another_profile(layout, monkeypatch):
    from hermes_cli.config_effective import _EFFECTIVE_CACHE, load_user_config_effective
    root, profile = layout
    monkeypatch.setenv("HERMES_HOME", str(root))
    _EFFECTIVE_CACHE.clear()
    eff = load_user_config_effective(profile / "config.yaml")
    assert eff["agent"] == {"reasoning_effort": "high", "max_turns": 77}
    assert eff["display"]["skin"] == "global-skin"
    assert "gateway" not in eff and "_config_version" in eff  # profile's own version, host keys dropped
    assert "max_turns" not in base_layer.read_layered_config_raw(profile / "config.yaml")
    assert base_layer.read_layered_config_raw(profile / "config.yaml")["agent"]["max_turns"] == 77
    assert "display" not in load_user_config_effective(root / "config.yaml").get("agent", {})
    text = (root / "config.yaml").read_text().replace("global-skin", "new-skin")
    _write(root / "config.yaml", text)
    os.utime(root / "config.yaml", ns=(1, 1))
    assert load_user_config_effective(profile / "config.yaml")["display"]["skin"] == "new-skin"


def test_base_layer_never_discovers_plugins(layout, monkeypatch):
    # hermes_cli.config calls load_config at import time; plugin loader threads import that module.
    import hermes_cli.plugins as plugins_mod

    def _boom(*a, **k):
        raise AssertionError("base layer must not trigger plugin discovery")

    monkeypatch.setattr(plugins_mod, "discover_plugins", _boom)
    assert "telegram" in base_layer._channel_index(layout[0])[0]
    assert base_layer.is_channel_env("TELEGRAM_BOT_TOKEN", layout[0])
    assert not base_layer.is_channel_env("CURSOR_API_KEY", layout[0])
    assert not base_layer.is_channel_env("EMAIL_PASSWORD", layout[0])  # shared with mail skills
    config_mod.load_config()
    config_mod.load_env()


def test_strip_keeps_sections_empty_in_the_file():
    raw = {"agent": {"personalities": {}}, "gateway": {"profile_routes": [1]}, "x": {"y": {"z": 1}}}
    stripped = base_layer.strip_host_only(raw)
    assert stripped["agent"] == {"personalities": {}}
    assert "gateway" not in stripped
    assert stripped["x"] == {"y": {"z": 1}}


def test_channel_credentials_and_sections_stay_with_host(layout):
    root, _ = layout
    text = (root / "config.yaml").read_text() + "telegram:\n  reply_to_mode: all\nplatforms:\n  telegram: {enabled: true}\n"
    _write(root / "config.yaml", text)
    env = config_mod.load_env()
    assert "TELEGRAM_BOT_TOKEN" not in env and "DISCORD_BOT_TOKEN" not in env
    base = base_layer.load_base_config()
    assert "telegram" not in base and "platforms" not in base


def test_secret_scope_includes_base_without_allowlist(layout):
    from agent.secret_scope import build_profile_secret_scope

    _, profile = layout
    scope = build_profile_secret_scope(profile)
    assert scope["CURSOR_API_KEY"] == "global-key"
    assert scope["SHARED_TOKEN"] == "p"
    assert "TELEGRAM_ALLOWED_USERS" not in scope


def test_secret_scope_for_root_has_no_base_layer(layout):
    from agent.secret_scope import build_profile_secret_scope

    root, _ = layout
    scope = build_profile_secret_scope(root)
    assert scope["TELEGRAM_ALLOWED_USERS"] == "1,2"


def test_disable_switch(layout, monkeypatch):
    monkeypatch.setenv("HERMES_DISABLE_BASE_LAYER", "1")
    config_mod._LOAD_CONFIG_CACHE.clear()
    assert base_layer.base_home() is None
    assert config_mod.load_config()["display"].get("skin") != "global-skin"


def test_global_plugins_scanned_before_profile(layout):
    from hermes_cli import plugins_discovery

    root, profile = layout
    for home, label in ((root, "global"), (profile, "profile")):
        _write(home / "plugins" / "shared" / "plugin.yaml", f"name: shared\nversion: '1'\ndescription: {label}\n")
    _write(root / "plugins" / "only-global" / "plugin.yaml", "name: only-global\nversion: '1'\n")
    winners = plugins_discovery.resolve_manifest_winners(plugins_discovery.collect_directory_manifests())
    assert "only-global" in winners
    assert str(profile) in str(winners["shared"].path)


def test_config_set_global_writes_root(layout, capsys):
    from hermes_cli.config_global_cmd import global_home

    root, profile = layout
    with global_home():
        config_mod.set_config_value("display.skin", "via-global")
    assert "via-global" in (root / "config.yaml").read_text()
    assert "via-global" not in (profile / "config.yaml").read_text()


def test_origins(layout):
    from hermes_cli.config_global_cmd import config_origins

    origins = config_origins()
    assert origins["agent.reasoning_effort"][1] == "profile"
    assert origins["display.skin"][1] == "global"
