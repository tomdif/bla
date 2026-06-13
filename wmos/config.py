"""Config: defaults <- ~/.wmos/config.json (or $WMOS_CONFIG) <- environment overrides."""
import os, json

DEFAULTS = {
    "adapter": "grid",
    "autonomy": "manual",
    "model": "claude-haiku-4-5-20251001",
    "trust_threshold": 8.0,
    "storage": "~/.wmos",
    "host": "127.0.0.1",
    "port": 8765,
}

_ENV = {"WMOS_ADAPTER": "adapter", "WMOS_AUTONOMY": "autonomy", "WMOS_MODEL": "model",
        "WMOS_STORAGE": "storage", "WMOS_HOST": "host", "WMOS_PORT": "port"}


def load_config(path=None):
    cfg = dict(DEFAULTS)
    p = os.path.expanduser(path or os.environ.get("WMOS_CONFIG", "~/.wmos/config.json"))
    if os.path.exists(p):
        try:
            with open(p) as f: cfg.update(json.load(f))
        except (OSError, json.JSONDecodeError):
            pass
    for env, key in _ENV.items():
        if os.environ.get(env):
            v = os.environ[env]
            cfg[key] = int(v) if key == "port" else v
    return cfg
