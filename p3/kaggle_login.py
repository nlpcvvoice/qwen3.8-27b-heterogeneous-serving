"""Kaggle login helper (project-local, in-memory token).

Reads the access token from ./reference/API-Token into memory only.
Never prints the token value; expose masked length / ok-status only.

Usage:
    import kaggle_login as kl
    kl.login()                # read token + export KAGGLE_API_TOKEN / KAGGLE_CONFIG_DIR
    s = kl.auth_session()     # requests.Session with Bearer auth
    api = kl.get_kaggle_api() # official kaggle.api authenticated instance
    print(kl.verify())        # masked sanity check

Horizontal constraints (see MASTER.md):
    - token source : ./reference/API-Token            (read-only, memory only)
    - config dir   : ./tmp/kaggle_cfg                 (keeps writes inside project)
    - no writes to ~/.kaggle / ~/.kaggle/access_token
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent
_TOKEN_PATH = _BASE_DIR / "reference" / "API-Token"
_CFG_DIR = _BASE_DIR / "tmp" / "kaggle_cfg"

KAGGLE_API_TOKEN_ENV = "KAGGLE_API_TOKEN"
KAGGLE_CONFIG_DIR_ENV = "KAGGLE_CONFIG_DIR"


def _read_token() -> str:
    """Read the raw access token into memory. Never persists/logs the value."""
    if not _TOKEN_PATH.exists():
        raise FileNotFoundError(f"token file not found: {_TOKEN_PATH}")
    token = _TOKEN_PATH.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"empty token in {_TOKEN_PATH}")
    return token


def token() -> str:
    """Return the raw access token (for in-memory use only; do NOT print)."""
    return _read_token()


def login() -> None:
    """Load token into memory and export the env vars the Kaggle clients need."""
    tok = _read_token()
    _CFG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ[KAGGLE_API_TOKEN_ENV] = tok
    os.environ.setdefault(KAGGLE_CONFIG_DIR_ENV, str(_CFG_DIR))


def auth_session(max_retries: int = 3) -> "object":
    """Return a requests.Session pre-authenticated with a Bearer access token."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    login()
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {os.environ[KAGGLE_API_TOKEN_ENV]}"
    retry = Retry(
        total=max_retries,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def get_kaggle_api() -> "object":
    """Return an authenticated official kaggle.api.KaggleApi instance."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    login()
    api = KaggleApi()
    api.authenticate()
    return api


def verify(api: "object | None" = None) -> dict:
    """Perform a lightweight, masked check that credentials load correctly.

    Returns {"ok": bool, "token_prefix": str, "token_len": int, "status": str}
    """
    tok = _read_token()
    result = {
        "ok": bool(tok.startswith("KGAT_") and len(tok) > 10),
        "token_prefix": tok[:5],
        "token_masked": f"{tok[:5]}...{tok[-3:]}",
        "token_len": len(tok),
        "status": "token_loaded",
    }
    try:
        inst = api if api is not None else get_kaggle_api()
        result["api"] = getattr(inst, "_authenticated", False)
        result["status"] = "ok" if result["api"] else "loaded_not_authenticated"
        result["ok"] = result["ok"] and result["api"]
    except Exception as exc:  # noqa: BLE001 - surface status, never the token
        result["status"] = f"error: {type(exc).__name__}"
        result["ok"] = False
    return result


def register_service(engine: str, endpoint: str, model: str, api_key: str,
                     kernel: str = "", topic: str = "", **extra) -> None:
    """Atomically record a live OpenAI-compatible endpoint for other machines.

    Writes {engine: {endpoint, api_key, model, ...}} (merged) to
    ./tmp/current_services.json. Called automatically by the watchers /
    controller when an engine reaches READY.
    """
    if not endpoint or not api_key:
        return
    path = _BASE_DIR / "tmp" / "current_services.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "endpoint": endpoint,
        "api_key": api_key,
        "model": model,
        "kernel": kernel,
        "topic": topic,
        "ready_at": None,
        **extra,
    }
    try:
        import time
        record["ready_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
    except Exception:
        pass
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        data = {}
    data[engine] = record
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)
    path.chmod(0o600)
    print(f"[services] {engine} registered -> {path}")


if __name__ == "__main__":
    import pprint

    login()
    pprint.pprint(verify())
