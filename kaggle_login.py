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

import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent
_TOKEN_PATH = _BASE_DIR / "reference" / "API-Token"
_CFG_DIR = _BASE_DIR / "tmp" / "kaggle_cfg"


def _silence_sdk_auth_banner() -> None:
    """Deterministically silence the SDK's print_auth_help() banner.

    The kaggle package calls api.authenticate() at import time (kaggle/__init__);
    when the token-introspect call is flaky it falls through to print_auth_help()
    then exit, caught by its own try/except — but prints a noisy banner. We (1)
    set the env token before any kaggle import so auth succeeds when possible,
    (2) import the submodule with stdout/stderr redirected so any leftover banner
    is dropped, and (3) patch the single print site to a no-op for later paths.
    The from-import form is reliable across kaggle 2.2.x; the plain module-form
    triggers an importlib quirk that can skip the patch.
    """
    try:
        os.environ.setdefault(KAGGLE_API_TOKEN_ENV, _read_token())
        os.environ.setdefault(KAGGLE_CONFIG_DIR_ENV, str(_CFG_DIR))
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            # from-import reliably sets sys.modules["kaggle.api.kaggle_api_extended"]
            # and avoids a 3.12 + kaggle 2.2.x import bug with the plain `import` form.
            from kaggle.api.kaggle_api_extended import KaggleApi as _probe  # noqa: F401
        kae = sys.modules.get("kaggle.api.kaggle_api_extended")
        if kae is not None and not getattr(kae, "_auth_help_silenced", False):
            kae.print_auth_help = lambda: None  # type: ignore[assignment]
            kae._auth_help_silenced = True
    except FileNotFoundError:
        pass  # token file absent; callers see the error on login()
    except Exception:
        pass


_silence_sdk_auth_banner()

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
    """Return an authenticated official kaggle.api.KaggleApi instance.

    Import + authenticate are wrapped in stdout/stderr redirects so the SDK's
    "Authentication required" banner can never leak into our output, and a
    SystemExit from authenticate() (its fallback path) can't kill the caller.
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
        # Token-introspect failed (flaky) but the token itself is fine; make sure
        # downstream SDK calls (quota_view...) still send it.
        try:
            tok = os.environ[KAGGLE_API_TOKEN_ENV]
            api.config_values["token"] = tok
            api.config_values["username"] = api.config_values.get("username", "tentenshishi")
            api.config_values["auth_method"] = "access_token"
        except Exception:  # noqa: BLE001
            pass
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


def quota() -> dict:
    """Return current weekly GPU/TPU accelerator quota (hours + refresh time).

    Uses the official SDK's quota_view() — no token data exposed. Example:
        {"gpu":  {"used_h": .., "remaining_h": .., "total_h": ..},
         "tpu":  {...}, "refresh_at": "..."}
    """
    api = get_kaggle_api()
    # The SDK prints an "Authentication required" help banner to stdout when it
    # cannot find a kaggle.json even though we authenticate via env token. Swallow
    # the noise; the quota data itself is what we return.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        resp = api.quota_view()
    out = {"refresh_at": str(resp.quota_refresh_time)}
    for name in ("gpu_quota", "tpu_quota"):
        q = getattr(resp, name)
        if q is None:
            out[name.replace("_quota", "")] = None
            continue
        used = q.time_used.total_seconds() / 3600
        total = q.total_time_allowed.total_seconds() / 3600
        out[name.replace("_quota", "")] = {
            "used_h": round(used, 2),
            "remaining_h": round(max(0.0, total - used), 2),
            "total_h": round(total, 2),
        }
    return out


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


def _machine_kind(machine_shape: str, enable_gpu: bool, enable_tpu: bool) -> str:  # noqa: D401
    """Classify kernel into 'gpu'/'tpu'/'cpu' from metadata booleans + machine_shape."""
    ms = (machine_shape or "").lower()
    if enable_tpu or "tpu" in ms:
        return "tpu"
    if enable_gpu or "gpu" in ms or "tesla" in ms or "a100" in ms or "p100" in ms or "k80" in ms:
        return "gpu"
    return "cpu"


def running_jobs(max_age_h: float = 36.0) -> dict:
    """Snapshot of my recent kernels: status + machine kind + quota.

    Only looks at kernels with last_run_time within `max_age_h` hours (default 36h
    to catch keepalive sessions but skip stale runs). Returns dict with keys
    "jobs" (list of dicts), "quota" (quota()), "refresh_at".
    """
    import time
    from concurrent.futures import ThreadPoolExecutor
    api = get_kaggle_api()
    cutoff_ts = time.time() - max_age_h * 3600

    ks = api.kernels_list(mine=True)
    recent = []
    for k in (ks or []):
        try:
            lr = k.last_run_time  # ISO string
            if lr:
                from datetime import datetime, timezone
                ts = datetime.fromisoformat(lr.replace("Z", "+00:00")).timestamp()
                if ts < cutoff_ts:
                    continue
        except Exception:
            pass
        recent.append(k)

    def _status(ref: str) -> str:
        try:
            st = api.kernels_status(ref)
            fm = st.to_field_map() if hasattr(st, "to_field_map") else (st or {})
            return fm.get("status", "unknown")
        except Exception:
            return "unknown"

    def _probe(ref: str) -> dict:
        status = "unknown"
        kind = machine_shape = ""
        status = _status(ref)
        if status in ("RUNNING", "QUEUED"):
            try:
                import kagglesdk.kernels.services.kernels_api_service as kms
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    with api.build_kaggle_client() as kaggle:
                        r = kms.ApiGetKernelRequest()
                        r.user_name, r.kernel_slug = ref.split("/", 1)
                        r.version_label = None
                        md = kaggle.kernels.kernels_api_client.get_kernel(r).metadata.to_field_map()
                machine_shape = md.get("machine_shape", "")
                kind = _machine_kind(machine_shape, bool(md.get("enable_gpu")), bool(md.get("enable_tpu")))
            except Exception:
                pass
        return {"ref": ref, "status": status, "kind": kind, "machine_shape": machine_shape}

    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = list(pool.map(lambda k: _probe(k.ref if hasattr(k, 'ref') else str(k)), recent))

    # Also print a compact table
    if jobs:
        print(f"{'ref':<40} {'status':<10} {'kind':<5} {'machine_shape':<16}")
        print("-" * 75)
        for j in jobs:
            print(f"{j['ref']:<40} {j['status']:<10} {j['kind']:<5} {j['machine_shape']:<16}")

    q = quota()
    print(f"\nquota: gpu used {q['gpu']['used_h']:.2f}h / remaining {q['gpu']['remaining_h']:.2f}h | "
          f"tpu used {q['tpu']['used_h']:.2f}h / remaining {q['tpu']['remaining_h']:.2f}h | refresh {q['refresh_at']}")
    return {"jobs": jobs, "quota": q, "refresh_at": q.get("refresh_at")}


def stop_gpu_job(kernel_ref: str, topic: str | None = None, timeout_s: int = 360) -> dict:
    """Stop a running GPU job via the CPU-free stopper kernel (CONTROL-stop over ntfy).

    `topic` is the ntfy topic the kernel listens on. If omitted, it is looked up
    from tmp/kaggle-gpu-lab.json (for our serve kernel) or tmp/current_services.json.
    If no topic is found, returns {stopped: False, reason: ...} — not an exception,
    so callers can decide gracefully.

    The stopper itself is a tiny free CPU kernel (no GPU quota consumed) that posts
    CONTROL-stop repeatedly over ~4 minutes on Kaggle's internal network.
    Returns dict with keys: stopped, status, reason.
    """
    login()
    if topic is None:
        topic = _find_control_topic(kernel_ref)
    if not topic:
        return {"stopped": False, "reason": "no control channel (ntfy topic) known for this kernel",
                "hint": "pass the ntfy topic explicitly, or register it via register_service()"}

    # Check if already COMPLETE
    try:
        cur = _status_of(get_kaggle_api(), kernel_ref)
        if cur in ("COMPLETE", "ERROR"):
            return {"stopped": True, "status": cur, "reason": "already done"}
    except Exception:
        pass

    # Build + push the stopper kernel with this topic
    import tempfile
    from pathlib import Path as _P
    ctrl_src = (_BASE_DIR / "p3" / "kernel" / "ctrl_stop.py").read_text()
    ctrl_src = ctrl_src.replace('TOPIC = None  # __STOPPER_TOPIC__', f'TOPIC = {topic!r}', 1)
    slug = f"gpu-stop-{abs(hash(kernel_ref + topic)) & 0xFFFFFF:06x}"
    with tempfile.TemporaryDirectory() as td:
        td = _P(td)
        (td / "ctrl_stop.py").write_text(ctrl_src)
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{_read_username()}/{slug}",
            "title": slug,
            "code_file": "ctrl_stop.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false", "enable_tpu": "false", "enable_internet": "true",
            "machine_shape": "CPU",
            "dataset_sources": [], "competition_sources": [],
            "kernel_sources": [], "model_sources": [],
        }, indent=1))
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(td)],
                           capture_output=True, text=True)
        if "successfully pushed" not in ((r.stdout or "") + (r.stderr or "")):
            return {"stopped": False, "reason": "stopper push failed",
                    "detail": ((r.stdout or "") + (r.stderr or ""))[-400:]}

    import time as _time
    _time.sleep(12)  # let stopper run a few post cycles
    deadline = _time.time() + timeout_s
    while _time.time() < deadline:
        try:
            cur = _status_of(get_kaggle_api(), kernel_ref)
            if cur in ("COMPLETE", "ERROR"):
                return {"stopped": True, "status": cur, "reason": "stopped via CONTROL-stop"}
        except Exception:
            pass
        _time.sleep(10)
    return {"stopped": False, "reason": "timed out waiting for kernel to stop", "timeout_s": timeout_s}


def _status_of(api: "object", ref: str) -> str:
    """Return the kernels_status string ('RUNNING'/'COMPLETE'/...) for a kernel ref."""
    try:
        st = api.kernels_status(ref)
        fm = st.to_field_map() if hasattr(st, "to_field_map") else (st or {})
        return fm.get("status", "unknown")
    except Exception:
        return "unknown"


def _find_control_topic(kernel_ref: str) -> str | None:
    """Best-effort lookup of the ntfy topic for a kernel from known state files."""
    # 1. tmp/kaggle-gpu-lab.json (single-topic for our serve kernel)
    state_file = _BASE_DIR / "tmp" / "kaggle-gpu-lab.json"
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            if state.get("kernel") == kernel_ref and state.get("topic"):
                return state["topic"]
        except Exception:
            pass
    # 2. tmp/current_services.json (register_service entries: keyed by engine name)
    svc_file = _BASE_DIR / "tmp" / "current_services.json"
    if svc_file.exists():
        try:
            for rec in json.loads(svc_file.read_text()).values():
                if rec.get("kernel") == kernel_ref and rec.get("topic"):
                    return rec["topic"]
        except Exception:
            pass
    return None


def _read_username() -> str:
    """Return the Kaggle username for kernel metadata IDs."""
    try:
        return get_kaggle_api().config_values.get("username", "tentenshishi")
    except Exception:
        return "tentenshishi"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("kaggle_login", description="Kaggle account helpers (masked credentials).")
    ap.add_argument("cmd", nargs="?", choices=["jobs", "stop", "verify", "quota"], default="verify",
                    help="jobs=snapshot of recent kernels+quota | stop <kernel>=stop a running GPU job")
    ap.add_argument("target", nargs="?", default="", help="kernel ref for 'stop' (owner/slug)")
    args = ap.parse_args()

    login()
    if args.cmd == "jobs":
        running_jobs()
    elif args.cmd == "stop":
        if "/" not in args.target:
            sys.exit("usage: python kaggle_login.py stop <owner/slug>")
        res = stop_gpu_job(args.target)
        print(json.dumps({k: v for k, v in res.items()}, indent=1))
    elif args.cmd == "quota":
        for res, v in quota().items():
            if res == "refresh_at":
                print("refresh_at:", v)
                continue
            if v is None:
                print(f"  {res:6s} n/a")
                continue
            print(f"  {res:6s} used {v['used_h']:6.2f}h | remaining {v['remaining_h']:6.2f}h | total {v['total_h']:.2f}h")
    else:
        api = get_kaggle_api()
        print("verify :", verify(api))
        print()
        for res, v in quota().items():
            if res == "refresh_at":
                print("quota  : (weekly, refresh", v, ")")
                continue
            if v is None:
                print(f"  {res:6s} n/a")
                continue
            print(f"  {res:6s} used {v['used_h']:6.2f}h | remaining {v['remaining_h']:6.2f}h | total {v['total_h']:.2f}h")
