#!/usr/bin/env python3
"""Register the current live endpoint/key for a running engine.

Covers the `--no-watch` flow (push-only, controller path) and any after-the-fact
need: reads tmp/<engine>-lab.json (kernel/topic/api_key), scans the ntfy topic
for the latest endpoint, then atomically writes it into tmp/current_services.json
so client machines can simply read that file.

Usage:
    ./Venv/bin/python app/register_service.py take tpu|gpu [--force]
    ./Venv/bin/python app/register_service.py status
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import kaggle_login as kl

DEFAULT_MODEL = "qwen3.8-27b"


def _endpoint_from_topic(topic):
    """Return (endpoint, model, ctx) by scanning ntfy for tunnel-url/serving/ready/heartbeat."""
    ep = model = ctx = None
    try:
        with urllib.request.urlopen(
                f"https://ntfy.sh/{topic}/json?poll=1&since=0", timeout=20) as r:
            body = r.read().decode()
    except Exception as exc:
        print(f"  ! ntfy poll failed: {exc}")
        return None, None, None
    for line in body.splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "message":
            continue
        try:
            ev = json.loads(e.get("message", "{}"))
        except Exception:
            continue
        phase = ev.get("phase")
        if phase == "tunnel-url":
            ep = ev.get("endpoint") or ep
        elif phase in ("serving", "ready"):
            ep = ev.get("endpoint") or ep
            model = ev.get("model") or model
            ctx = ev.get("ctx_size") or ev.get("max_model_len") or ctx
        elif phase == "heartbeat":
            ep = ev.get("endpoint") or ep
    return ep, model, ctx


def take(engine, force=False):
    state_file = ROOT / "tmp" / f"kaggle-{engine}-lab.json"
    if not state_file.exists():
        print(f"no state file for '{engine}' (was it pushed from this dir?): {state_file}")
        sys.exit(1)
    d = json.loads(state_file.read_text())
    kernel, topic, api_key = d.get("kernel"), d.get("topic"), d.get("api_key")
    if not (topic and api_key):
        print(f"incomplete state for '{engine}' (need topic+api_key): {d}")
        sys.exit(1)
    print(f"engine={engine} kernel={kernel}\n  topic={topic}\n  scanning ntfy for endpoint...")
    ep, model, ctx = _endpoint_from_topic(topic)
    if not ep:
        print("  no endpoint event yet (still provisioning?) — rerun take once it's live.")
        sys.exit(1)
    kl.register_service(engine, ep, model or DEFAULT_MODEL, api_key,
                        kernel=kernel, topic=topic,
                        ctx_size=ctx, source="register_service")
    print(f"  endpoint: {ep}\n  model   : {model or DEFAULT_MODEL}")


def status():
    path = ROOT / "tmp" / "current_services.json"
    if not path.exists():
        print("no services registered yet (tmp/current_services.json missing)")
        sys.exit(0)
    data = json.loads(path.read_text())
    for engine, rec in data.items():
        key = rec.get("api_key", "")
        masked = f"{key[:6]}...{key[-4:]}" if key else "(none)"
        print(f"{engine:5s} {rec.get('ready_at', '?'):>22}  {rec['endpoint']}  key={masked}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("take", "status"))
    ap.add_argument("engine", nargs="?", choices=("tpu", "gpu"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.cmd == "status":
        status()
    else:
        take(args.engine, force=args.force)


if __name__ == "__main__":
    main()