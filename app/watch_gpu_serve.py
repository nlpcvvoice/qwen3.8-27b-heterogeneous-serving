#!/usr/bin/env python3
"""Monitor the GPU serve kernel via ntfy events + kaggle status. Prints READY
banner with endpoint/key (from the local state file, never the Kaggle token)."""
import json, re, subprocess, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kaggle_login as kl

ROOT = Path(__file__).resolve().parent.parent
STATE = json.loads((ROOT / "tmp" / "kaggle-gpu-lab.json").read_text())
KID = STATE["kernel"]
TOPIC = STATE["topic"]
ENDPOINT = None

kl.login()

PHASES = {
    "installed": "runtime ready", "weights-mounted": "weights from private dataset (no download)",
    "weights-download": "weights downloading from HF...", "tunnel-url": "endpoint reserved (not live)",
    "serving": "server healthy", "ready": "READY", "auto-shutdown": "stopped by keepalive",
    "stopped": "stopped", "failed": "FAILED", "benchmark": "self-test benchmark",
    "benchmark-error": "self-test error",
}


def read_events(since):
    try:
        with urllib.request.urlopen(f"https://ntfy.sh/{TOPIC}/json?poll=1&since={since}",
                                    timeout=15) as r:
            body = r.read().decode()
    except Exception:
        return []
    evs = []
    for line in body.splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "message":
            continue
        try:
            evs.append((e.get("id"), e["time"], json.loads(e.get("message", "{}"))))
        except Exception:
            continue
    return evs


_registered = {}


def render(ev):
    global ENDPOINT
    phase = ev.get("phase")
    if phase == "tunnel-url":
        ENDPOINT = ev.get("endpoint")
        print(f"  endpoint reserved: {ENDPOINT}")
    elif phase == "ready":
        print("\n" + "=" * 66)
        print("  GPU ENDPOINT IS LIVE")
        print(f"  base URL : {ev.get('endpoint')}")
        print(f"  API key  : {STATE['api_key']}")
        print(f"  model    : {ev.get('model')}  (ctx {ev.get('ctx_size')}, 2xT4)")
        print("=" * 66)
        print("Try it:")
        print(f"  curl {ev.get('endpoint')}/chat/completions -H 'Authorization: Bearer {STATE['api_key']}' \\")
        print("    -H 'Content-Type: application/json' -d '{" +
              '"model": "' + ev.get("model", "qwen3.8-27b") + '", "messages": [{"role": "user", "content": "Hello!"}]}')
    elif phase == "serving":
        ep = ENDPOINT or ev.get("endpoint")
        if ep and ep not in _registered:
            _registered[ep] = True
            kl.register_service("gpu", ep, "qwen3.8-27b", STATE["api_key"],
                                kernel=KID, topic=TOPIC,
                                ctx_size=ev.get("ctx_size"))
            print("  serving healthy — registered in tmp/current_services.json")
    elif phase == "benchmark":
        ep = ENDPOINT or ev.get("endpoint")
        if ep and ep not in _registered:
            _registered[ep] = True
            kl.register_service("gpu", ep, "qwen3.8-27b", STATE["api_key"],
                                kernel=KID, topic=TOPIC,
                                ctx_size=ev.get("ctx_size"))
            print("  benchmark passed — registered in tmp/current_services.json")
        print(f"  benchmark: {ev.get('decode_tok_s', '?')} tok/s  sanity={ev.get('sanity', '')[:40]!r}")
    elif phase == "vram":
        print(f"  VRAM ctx={ev.get('ctx_size')} np={ev.get('n_parallel')} KV={ev.get('kv')} "
              f"(llama lines: {ev.get('n_lines', 0)}, nvidia-smi rows: {ev.get('n_smi', 0)})")
        for ln in (ev.get("lines") or "").splitlines()[-8:]:
            print("    " + ln[-120:])
        for ln in (ev.get("smi") or [])[:4]:
            print("    smi   " + ln)
    elif phase == "engine-cached":
        print(f"  cached built binary: {ev.get('path')}  ({ev.get('size_gb', '?')} GiB) — pullable on kernel completion")
    elif phase == "weights-mounted":
        print("  " + PHASES.get(phase, phase))
    elif phase == "heartbeat":
        print(f"  heartbeat {ev.get('up_min', '?')} min — {ev.get('endpoint', ENDPOINT)}")
    elif phase in PHASES:
        label = PHASES[phase]
        extra = f"  ({json.dumps({k: v for k, v in ev.items() if k not in ('phase',)})})"
        print("  " + (label + extra if phase == "failed" else label))
    elif phase in ("failed", "stopped", "auto-shutdown"):
        print("  " + " ".join(f"{k}={v}" for k, v in ev.items())[:200])


since = int(time.time()) - 600
seen_ids = set()
seen = False
last_status = None
while True:
    for eid, ts, ev in read_events(since):
        since = max(since, ts)
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        seen = True
        render(ev)
        if ev.get("phase") in ("failed", "auto-shutdown", "stopped"):
            sys.exit(0)
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", KID],
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r'"KernelWorkerStatus\.(\w+)"', out)
    status = m.group(1) if m else "UNKNOWN"
    if status != last_status:
        if status == "QUEUED":
            print(f"{time.strftime('%H:%M:%S')} queued — waiting for a GPU T4 x2 slot...")
        elif status == "RUNNING" and not seen:
            print(f"{time.strftime('%H:%M:%S')} RUNNING — provisioning GPU session...")
        elif status in ("ERROR", "CANCELACKNOWLEDGED", "COMPLETE"):
            print(f"{time.strftime('%H:%M:%S')} kernel finished ({status}).")
            sys.exit(0)
        last_status = status
    time.sleep(30)