#!/usr/bin/env python3
"""Send a CONTROL/stop to the running kernel's ntfy topic (frees the GPU quota
immediately; the kernel os._exit(0)s and output becomes downloadable)."""
import json, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kit_config import STATE_FILE


def main():
    if not STATE_FILE.exists():
        sys.exit(f"state file not found: {STATE_FILE}\nrun push_job.py first")
    state = json.loads(STATE_FILE.read_text())
    topic = state["topic"]
    body = json.dumps({"phase": "CONTROL", "cmd": "stop"}).encode()
    req = urllib.request.Request(
        f"https://ntfy.sh/{topic}", data=body,
        headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=15)
        print(f"[t4-kit] CONTROL-stop sent to {topic}")
    except Exception as e:
        sys.exit(f"CONTROL-stop failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()