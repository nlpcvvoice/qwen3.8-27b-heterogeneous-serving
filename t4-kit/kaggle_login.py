"""Kaggle login helper (project-local, in-memory token) for the t4-kit.

Token source resolution (first hit wins):
  env KAGGLE_TOKEN_FILE
  <kit>/reference/API-Token
  <project-root>/reference/API-Token
Read into memory only; never print the token (masked len/prefix only).
Writes KAGGLE_CONFIG_DIR inside ./tmp/t4-kit — never touches ~/.kaggle.

Usage:
    import kaggle_login as kl
    kl.login()                # export KAGGLE_API_TOKEN / KAGGLE_CONFIG_DIR
    api = kl.get_kaggle_api() # official kaggle.api authenticated instance
    s   = kl.auth_session()   # requests.Session with Bearer auth
    print(kl.verify())        # masked sanity check
    print(kl.quota())         # weekly GPU quota (hours)
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

from kit_config import CFG_DIR, TOKEN_CANDIDATES

KAGGLE_API_TOKEN_ENV = "KAGGLE_API_TOKEN"
KAGGLE_CONFIG_DIR_ENV = "KAGGLE_CONFIG_DIR"


def _read_token() -> str:
    """Read the raw access token into memory. Never persists/logs the value."""
    for cand in TOKEN_CANDIDATES:
        if cand.exists():
            token = cand.read_text(encoding="utf-8").strip()
            if token:
                return token
    raise FileNotFoundError(
        "no Kaggle token found; set env KAGGLE_TOKEN_FILE or place reference/API-Token "
        "in <kit>/ or <project-root>/ (see README)")
    # noqa: PLR0204


def _silence_sdk_auth_banner() -> None:
    """Deterministically silence the SDK's print_auth_help() banner.

    The kaggle package calls api.authenticate() at import time (kaggle/__init__);
    when the token-introspect call is flaky it falls back to print_auth_help() then
    exits — caught by its own try/except but prints noise. We (1) set the env token
    before any kaggle import so auth succeeds, (2) import the submodule with
    stdout/stderr redirected so leftover banner is dropped, (3) patch the single
    print site to a no-op for later paths. The from-import form is reliable across
    kaggle 2.2.x; the plain module-form triggers an importlib quirk that can skip
    the patch.
    """
    try:
        os.environ.setdefault(KAGGLE_API_TOKEN_ENV, _read_token())
        os.environ.setdefault(KAGGLE_CONFIG_DIR_ENV, str(CFG_DIR))
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            from kaggle.api.kaggle_api_extended import KaggleApi as _probe  # noqa: F401
        kae = sys.modules.get("kaggle.api.kaggle_api_extended")
        if kae is not None and not getattr(kae, "_auth_help_silenced", False):
            kae.print_auth_help = lambda: None  # type: ignore[assignment]
            kae._auth_help_silenced = True
    except (FileNotFoundError, Exception):  # noqa: BLE001
        pass  # token absent / sdk absent: callers see the error on login()


_silence_sdk_auth_banner()


def token() -> str:
    """Return the raw access token (for in-memory use only; do NOT print)."""
    return _read_token()


def login() -> None:
    """Load token and export the env vars the Kaggle clients need."""
    tok = _read_token()
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ[KAGGLE_API_TOKEN_ENV] = tok
    os.environ.setdefault(KAGGLE_CONFIG_DIR_ENV, str(CFG_DIR))


def auth_session(max_retries: int = 3):
    """Return a requests.Session pre-authenticated with a Bearer access token."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    login()
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {os.environ[KAGGLE_API_TOKEN_ENV]}"
    retry = Retry(total=max_retries, backoff_factor=0.5,
                  status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def get_kaggle_api():
    """Return an authenticated official kaggle.api.KaggleApi instance.

    Import + authenticate are redirect-wrapped so the SDK banner can never leak,
    and a SystemExit from authenticate() (its fallback path) can't kill us.
    """
    login()
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        try:
            api.authenticate()
        except BaseException:  # noqa: BLE001 - never let SDK exit() kill us
            pass
    if not getattr(api, "_authenticated", False):
        try:
            tok = os.environ[KAGGLE_API_TOKEN_ENV]
            api.config_values["token"] = tok
            api.config_values["username"] = api.config_values.get("username", "tentenshishi")
            api.config_values["auth_method"] = "access_token"
        except Exception:  # noqa: BLE001
            pass
    return api


def verify(api=None) -> dict:
    """Masked health check — never exposes the token value."""
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
    except Exception as exc:  # noqa: BLE001
        result["status"] = f"error: {type(exc).__name__}"
        result["ok"] = False
    return result


def quota() -> dict:
    """Return weekly GPU quota in hours + refresh time (T4 set only, no TPU)."""
    api = get_kaggle_api()
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        resp = api.quota_view()
    out = {"refresh_at": str(resp.quota_refresh_time), "gpu": None}
    q = getattr(resp, "gpu_quota", None)
    if q is not None:
        used = q.time_used.total_seconds() / 3600
        total = q.total_time_allowed.total_seconds() / 3600
        out["gpu"] = {"used_h": round(used, 2),
                      "remaining_h": round(max(0.0, total - used), 2),
                      "total_h": round(total, 2)}
    return out


def register_service(engine, endpoint, model, api_key, kernel="", topic="", **extra) -> None:
    """Record a live OpenAI-compatible endpoint in <project>/tmp/t4-kit/services.json."""
    if not endpoint or not api_key:
        return
    import json
    import time
    path = CFG_DIR.parent / "services.json"
    record = {"endpoint": endpoint, "api_key": api_key, "model": model,
              "kernel": kernel, "topic": topic, "ready_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              **extra}
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except Exception:  # noqa: BLE001
        data = {}
    data[engine] = record
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)
    path.chmod(0o600)
    print(f"[services] {engine} registered -> {path}")


if __name__ == "__main__":
    login()
    api = get_kaggle_api()
    print("verify :", verify(api))
    print("quota  : (weekly, refresh", quota()["refresh_at"], ")")
    g = quota()["gpu"]
    if g is None:
        print("  gpu n/a")
    else:
        print(f"  gpu used {g['used_h']:6.2f}h | remaining {g['remaining_h']:6.2f}h | total {g['total_h']:.2f}h")