#!/usr/bin/env python3
"""One-command Q4_K_M serve on 2xT4 + generate an opencode-ready config.

  python app/start_q4km.py          # push (cache-hot) kernel, wait for READY, write tmp/opencode-q4km.json
  python app/start_q4km.py --stop   # stop the running serve kernel via CPU stopper (free GPU)

Config mirrors the repo's opencode.json provider/kaggle shape (openai-compatible).
Secrets never printed; only lengths/masks shown.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import kaggle_login as kl

USER = "tentenshishi"
SERVE = f"{USER}/qwen38-gpu-serve"
STATE = ROOT / "tmp" / "kaggle-gpu-lab.json"
OUT_JSON = ROOT / "tmp" / "opencode-q4km.json"


def _run(py_script: str) -> None:
    r = subprocess.run([sys.executable, str(ROOT / py_script)], capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    print(out[-800:])
    if r.returncode != 0:
        sys.exit(f"{py_script} failed")


def _mask(s: str) -> str:
    return s[:14] + "..." + s[-4:] if len(s) > 22 else "***"


def wait_for_endpoint(timeout_s: int = 1500) -> tuple[str, str]:
    """Poll the kernel log stream (SSE, works from here even though ntfy.sh is blocked)
    until PHASE tunnel-url gives the public endpoint and PHASE ready confirms the server."""
    s = kl.auth_session()
    url = f"https://www.kaggle.com/api/v1/kernels/logs/stream/{SERVE}"
    endpoint = api_key = None
    deadline = time.time() + timeout_s
    t0 = time.time()
    while time.time() < deadline:
        try:
            with s.get(url, stream=True, timeout=60,
                       headers={"Accept": "text/event-stream, */*"}) as r:
                if r.status_code != 200:
                    time.sleep(10)
                    continue
                for raw in r.iter_lines(decode_unicode=True):
                    if time.time() > deadline:
                        break
                    if not raw or not raw.startswith("data: {"):
                        continue
                    try:
                        ev = json.loads(raw[6:])
                    except Exception:
                        continue
                    d = ev.get("data", "")
                    if endpoint is None and "PHASE tunnel-url" in d:
                        m = re.search(r'"endpoint": "([^"]+)"', d)
                        if m:
                            endpoint = m.group(1)
                            print(f"[q4km] tunnel-url: {_mask(endpoint)}  ({int(time.time()-t0)}s)", flush=True)
                    if api_key is None and "PHASE ready" in d:
                        m = re.search(r'"api_key": "([^"]+)"', d)
                        if m:
                            api_key = m.group(1)
                    if endpoint and api_key:
                        return endpoint, api_key
        except Exception as e:  # transient stream drop -> reconnect
            print(f"[q4km] stream reconnect ({type(e).__name__})", flush=True)
            time.sleep(5)
    sys.exit("timed out waiting for kernel READY (check kernel status/logs)")


def write_config(endpoint: str, api_key: str) -> Path:
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "kaggle": {
                "name": "Kaggle 2xT4 Q4_K_M (live)",
                "npm": "@ai-sdk/openai-compatible",
                "options": {"baseURL": endpoint, "apiKey": api_key},
                "models": {
                    "qwen3.8-27b": {"name": "Qwen3.8-27B · Q4_K_M · 2xT4 (live)"},
                },
            },
        },
        "model": "kaggle/qwen3.8-27b",
        "instructions": ["./MASTER.md"],
    }
    OUT_JSON.write_text(json.dumps(cfg, indent=2) + "\n")
    return OUT_JSON


def main() -> None:
    kl.login()
    _run("app/push_gpu_serve.py")

    state = json.loads(STATE.read_text())
    print(f"[q4km] kernel={state['kernel']}  waiting for READY (cache-hot, ~5-8 min)...")
    endpoint, api_key = wait_for_endpoint()

    if api_key is None:
        api_key = state.get("api_key")
    if api_key is None:
        sys.exit("api_key not found")
    out = write_config(endpoint, api_key)
    print(f"[q4km] config written -> {out}")
    print(f"[q4km] live endpoint     {_mask(endpoint)}")
    print(f"[q4km] api_key           {_mask(api_key)}")
    print("[q4km] use: opencode --config tmp/opencode-q4km.json   |   stop: python app/start_q4km.py --stop")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", action="store_true", help="stop the running serve kernel (free GPU)")
    args = ap.parse_args()
    if args.stop:
        _run("app/push_stopper.py")
        sys.exit(0)
    main()