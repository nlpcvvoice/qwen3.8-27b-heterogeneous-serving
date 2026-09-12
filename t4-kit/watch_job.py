#!/usr/bin/env python3
"""T4-job watcher: poll the launch ntfy topic + Kaggle SDK status.

Modes (--stop-on):
  built    stop kernel right after engine-built/engine-cached (cache-fill runs)
  serve    never auto-stop; exit only on COMPLETE/ERROR (normal serving)
Phases are always logged to <project>/tmp/t4-kit/watch.log.
Designed to be launched with `setsid` so it survives terminal close."""
import argparse, json, os, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kit_config import STATE_FILE, WATCH_LOG, RUNTIME
import kaggle_login as kl  # noqa: E402


def out(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(WATCH_LOG, "a") as f:
        f.write(line + "\n")


def poll_phase(since, topic):
    evs = []
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}/json?poll=1&since={since}",
            headers={"Accept": "text/event-stream"})
        with urllib.request.urlopen(req, timeout=40) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("{"):
                    continue
                ev = json.loads(line)
                since = max(since, ev.get("time", since))
                if ev.get("event") != "message":
                    continue
                msg = json.loads(ev.get("message", "{}"))
                evs.append({"t": ev.get("time"), "phase": msg.get("phase", ""),
                            "note": str(msg.get("note", ""))[:140]})
    except Exception as e:
        out(f"(ntfy poll error: {e.__class__.__name__})")
    return evs, since


def post_control_stop(topic):
    body = json.dumps({"phase": "CONTROL", "cmd": "stop"}).encode()
    req = urllib.request.Request(
        f"https://ntfy.sh/{topic}", data=body,
        headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=15)
        return True
    except Exception as e:
        out(f"(CONTROL-stop post failed: {e.__class__.__name__})")
        return False


def kernel_status(kernel_ref):
    try:
        api = kl.get_kaggle_api()
        r = api.kernels_status(kernel_ref)
        st = str(getattr(r, "status", "?"))
        return st
    except Exception as e:
        return f"err:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-on", choices=["built", "serve"], default="serve",
                    help="built = stop after engine-cached; serve = never auto-stop")
    ap.add_argument("--timeout-min", type=int, default=120,
                    help="watcher exits after this many minutes regardless")
    args = ap.parse_args()

    if not STATE_FILE.exists():
        sys.exit(f"state file not found: {STATE_FILE}\nrun push_job.py first")

    RUNTIME.mkdir(parents=True, exist_ok=True)
    WATCH_LOG.write_text("")
    state = json.loads(STATE_FILE.read_text())
    topic = state["topic"]
    kernel = state["kernel"]
    out(f"watching {kernel}  topic={topic}  stop-on={args.stop_on}")

    t0 = time.time()
    since = int(t0)
    stopped = False
    phases_seen = set()

    while time.time() - t0 < args.timeout_min * 60:
        evs, since = poll_phase(since, topic)
        for ev in evs:
            ph = ev["phase"]
            if ph not in phases_seen:
                phases_seen.add(ph)
                out(f"phase: {ph}  {ev['note']}")
        # built mode: stop once the binary lands in /kaggle/working
        if args.stop_on == "built" and not stopped and "engine-cached" in phases_seen:
            time.sleep(3)
            out("engine-cached -> sending CONTROL-stop")
            ok = post_control_stop(topic)
            out(f"CONTROL-stop sent: {ok}")
            stopped = True
        if {"failed", "engine-build-failed"} & phases_seen:
            out("BUILD FAILED -> exiting watcher")
            return 1
        st = kernel_status(kernel).lower()
        if stopped or args.stop_on == "serve":
            if "complete" in st or "done" in st:
                out(f"kernel DONE ({st})")
                return 0
            if any(k in st for k in ("error", "failed", "cancel")):
                out(f"kernel ended: {st}")
                return 1
        if "running" in st:
            out(f"kernel RUNNING (after {int(time.time()-t0)}s)")
        time.sleep(20)
    out("timeout hit; exiting watcher")
    return 1


if __name__ == "__main__":
    sys.exit(main())
