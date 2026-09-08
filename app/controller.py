#!/usr/bin/env python3
"""Local scheduling controller for the TPU+GPU heterogeneous serving (P1).

Implements the eco-mode policy decided on 2026-09-08:
  - GPU (2xT4) is the fast-start engine: on a cold start it serves first
  - controller monitors the TPU; once the TPU is READY and health-checked,
    auto primary switches to TPU and the GPU kernel is shut down (quota saved)
  - when the TPU session approaches its max duration, the GPU is pre-warmed
  - while serving on GPU, whenever a fresh TPU window is available, start the
    TPU immediately and switch back once verified healthy
  - global kill switch: power=off cancels both kernels (remote CONTROL-stop,
    the kernels shut down on the same ntfy channel we already use)

Router coupling: writes tmp/controller_state.json {"power", "primary"} which
the router reads per request (app/router.py pick()). Dry-run mode computes the
state machine but never pushes/cancels kernels.

API (127.0.0.1:8091):
  GET  /admin/status                  full state
  POST /admin/power   {"power":bool}  global kill switch
  POST /admin/action  {"act":...}     manual start-tpu/stop-tpu/start-gpu/stop-gpu
  POST /admin/mode    {"mode":"dry"|"live"}
Run: ./Venv/bin/python app/controller.py [--mode dry|live] [--port 8091]
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
CTL_STATE = ROOT / "tmp" / "controller_state.json"
APP = ROOT / "app"

ENGINES = {"tpu": {"file": ROOT / "tmp" / "kaggle-tpu-lab.json",
                   "pusher": [sys.executable, str(APP / "run_launch.py"), "serve",
                              "--no-watch"],
                   "status": "down", "endpoint": None, "healthy": False,
                   "since": 0, "ready_ts": None, "start_ts": None},
           "gpu": {"file": ROOT / "tmp" / "kaggle-gpu-lab.json",
                   "pusher": [sys.executable, str(APP / "push_gpu_serve.py")],
                   "status": "down", "endpoint": None, "healthy": False,
                   "since": 0, "ready_ts": None, "start_ts": None}}

TERMINAL = ("failed", "stopped", "auto-shutdown", "complete")

app = FastAPI(title="qwen3.8-27b controller")
state = {"power": True, "primary": None, "mode": "dry",
         "tpu": ENGINES["tpu"], "gpu": ENGINES["gpu"],
         "reason": "init", "loop_ts": 0, "events": {}}
lock = threading.Lock()

POLL = 20
STOPPED_GRACE = 30  # s before 'stopping' decays to 'down' when no event arrives
PROTECT_TPU = os.environ.get("KTL_PROTECT_TPU", "1") != "0"  # dev rule: never kill the live TPU
SIM_FILE = ROOT / "tmp" / "ctl_sim.json"
TPU_MAX_MIN = int(os.environ.get("KTL_TPU_MAX_MIN", "480"))
TPU_PREWARM_MIN = int(os.environ.get("KTL_TPU_PREWARM_MIN", "40"))
TPU_RETRY_MIN = int(os.environ.get("KTL_TPU_RETRY_MIN", "10"))
PROC = {"tpu": None, "gpu": None}


# ---------------------------------------------------------------- event I/O
def read_events(topic, since):
    try:
        with urllib.request.urlopen(
                f"https://ntfy.sh/{topic}/json?poll=1&since={since}", timeout=15) as r:
            body = r.read().decode()
    except Exception:
        return [], since
    evs, last = [], since
    for line in body.splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "message":
            continue
        last = max(last, e.get("time", since))
        try:
            evs.append(e["time"], json.loads(e.get("message", "{}")))
        except Exception:
            continue
    return evs, last


def publish(topic, msg):
    try:
        body = {"topic": topic, "title": "controller",
                "message": json.dumps(msg)}
        req = urllib.request.Request("https://ntfy.sh", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"(publish failed: {e})")


def healthy(eng):
    if not eng["endpoint"] or not eng.get("api_key"):
        return False
    try:
        req = urllib.request.Request(f"{eng['endpoint']}/models",
                                     headers={"Authorization": "Bearer " + eng["api_key"]})
        with urllib.request.urlopen(req, timeout=8) as r:
            return r.status == 200
    except Exception:
        return False


# ------------------------------------------------------------- engine actions
def read_registered():
    for key, eng in ENGINES.items():
        try:
            d = json.loads(eng["file"].read_text())
            eng["kernel"], eng["topic"] = d.get("kernel"), d.get("topic")
            eng["api_key"] = d.get("api_key")
        except Exception:
            pass


def launch(key):
    eng = ENGINES[key]
    if eng["status"] in ("starting", "ready") or PROC[key]:
        return
    if state["mode"] != "live":
        print(f"[dry] would start {key}: {' '.join(eng['pusher'])}")
        eng["status"] = "starting"
        eng["start_ts"] = time.time()
        return
    logf = open(ROOT / "tmp" / f"ctl-{key}.log", "a")
    PROC[key] = subprocess.Popen(eng["pusher"], stdout=logf, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT))
    eng["status"] = "starting"
    eng["start_ts"] = time.time()
    print(f"[live ] started {key} (pid {PROC[key].pid})")


def stop(key):
    eng = ENGINES[key]
    if eng["status"] == "down":
        return
    if key == "tpu" and PROTECT_TPU and not eng.get("force"):
        print(f"[guard ] TPU is protected (KTL_PROTECT_TPU=1) — stop blocked; "
              "set KTL_PROTECT_TPU=0 or use action-stoptpu-force to override")
        return
    topic = eng.get("topic")
    eng["status"] = "stopping"
    eng["stop_ts"] = time.time()
    if state["mode"] != "live":
        print(f"[dry] would CONTROL-stop {key} (topic={topic})")
        return
    if topic:
        publish(topic, {"phase": "CONTROL", "cmd": "stop", "by": "controller"})
    print(f"[live ] CONTROL-stop sent to {key}")


def _sim():
    """Dev workaround: pretend an engine is down without touching the real one."""
    try:
        return json.loads(SIM_FILE.read_text())
    except Exception:
        return {}


# ------------------------------------------------------------------- loop
def harvest_for(key, evs):
    eng = ENGINES[key]
    for ts, ev in evs:
        phase = ev.get("phase", "?")
        eng["since"] = max(eng["since"], ts)
        state["events"][phase] = state["events"].get(phase, 0) + 1
        if phase == "tunnel-url":
            eng["endpoint"] = ev.get("endpoint") or eng.get("endpoint")
        elif phase == "ready":
            eng["status"] = "ready"
            eng["endpoint"] = ev.get("endpoint") or eng.get("endpoint")
            eng["ready_ts"] = time.time()
            print(f"[event] {key} READY: {eng['endpoint']}")
        elif phase in TERMINAL:
            eng["status"] = "down"
            eng["endpoint"] = None
            eng["ready_ts"] = None
            PROC[key] = None
            print(f"[event] {key} -> {phase}")


def decide():
    t, g = ENGINES["tpu"], ENGINES["gpu"]
    sim = _sim()
    if sim.get("tpu"):
        # dev-only workaround: schedule as if the TPU were down (real TPU untouched)
        if g["status"] == "down":
            launch("gpu")
        state["primary"] = "gpu" if g["status"] in ("starting", "ready") else None
        return "tpu-sim-down (failover workaround)"
    reason = state["reason"]
    if not state["power"]:
        state["primary"] = None
        if t["status"] not in ("down", "stopping"):
            stop("tpu")
        if g["status"] not in ("down", "stopping"):
            stop("gpu")
        return "power-off"
    # fast engine first on cold start
    if g["status"] == "down" and t["status"] == "down":
        launch("gpu")
        state["primary"] = "gpu"
        return "cold: gpu-first"
    # TPU preferred whenever it is ready and healthy
    if t["status"] in ("starting", "ready"):
        if t["status"] == "ready" and t["healthy"]:
            if state["primary"] != "tpu":
                state["primary"] = "tpu"
                print("[sched ] primary -> tpu")
            if g["status"] in ("starting", "ready"):
                stop("gpu")
            # pre-warm GPU before the TPU session runs out
            if t["ready_ts"] and (time.time() - t["ready_ts"]) / 60 > TPU_MAX_MIN - TPU_PREWARM_MIN \
                    and g["status"] == "down":
                launch("gpu")
            return "tpu-serving"
        if state["primary"] != "tpu" and g["status"] == "ready":
            state["primary"] = "tpu"
        return "tpu-starting"
    # TPU down: serve on GPU, and grab a fresh TPU window when available
    if t["status"] == "down" and t["start_ts"] and \
            (time.time() - t["start_ts"]) / 60 > TPU_RETRY_MIN:
        launch("tpu")
        t["start_ts"] = None  # avoid retry spin
        return "tpu-retry"
    return reason


def probe():
    sim = _sim()
    for key, eng in ENGINES.items():
        eng["healthy"] = False if sim.get(key) else (
            healthy(eng) if eng["status"] == "ready" else False)
    if state["primary"] and not ENGINES[state["primary"]]["healthy"]:
        alt = "gpu" if state["primary"] == "tpu" else "tpu"
        if ENGINES[alt]["healthy"]:
            print(f"[sched ] {state['primary']} unhealthy -> failover to {alt}")
            state["primary"] = alt


def loop_once():
    with lock:
        read_registered()
        for key in ENGINES:
            eng = ENGINES[key]
            if eng["status"] == "stopping" and eng.get("stop_ts") and \
                    time.time() - eng["stop_ts"] > STOPPED_GRACE:
                eng["status"] = "down"   # dry-mode / missed-event fallback
                print(f"[event] {key} assumed down (grace expiry)")
            if eng.get("topic") and eng["status"] not in ("down", "stopping"):
                evs, en = read_events(eng["topic"], eng["since"])
                harvest_for(key, evs)
                eng["since"] = max(eng["since"], en)
        probe()
        state["reason"] = decide()
        state["loop_ts"] = time.time()
        CTL_STATE.write_text(json.dumps(
            {"power": state["power"], "primary": state["primary"],
             "reason": state["reason"], "ts": state["loop_ts"]}))
        print(f"[loop ] power={state['power']} primary={state['primary']} "
              f"tpu={ENGINES['tpu']['status']}/{ENGINES['tpu']['healthy']} "
              f"gpu={ENGINES['gpu']['status']}/{ENGINES['gpu']['healthy']} ({state['reason']})")


def bg():
    while True:
        try:
            loop_once()
        except Exception as e:
            print(f"[loop-err] {e}")
        time.sleep(POLL)


# ----------------------------------------------------------------- admin API
@app.post("/admin/sim")
async def sim(req: Request):
    d = await req.json()
    cur = _sim()
    for k in ("tpu", "gpu"):
        if k in d:
            cur[k] = bool(d[k])
    SIM_FILE.write_text(json.dumps(cur))
    return {"sim": cur, "note": "dev workaround only — real engines untouched"}


@app.get("/admin/status")
async def status():
    with lock:
        return {"power": state["power"], "primary": state["primary"],
                "reason": state["reason"], "mode": state["mode"],
                "tpu": {k: ENGINES["tpu"][k] for k in
                        ("status", "endpoint", "healthy", "ready_ts", "since")},
                "gpu": {k: ENGINES["gpu"][k] for k in
                        ("status", "endpoint", "healthy", "ready_ts", "since")},
                "events": state["events"]}


@app.post("/admin/power")
async def power(req: Request):
    d = await req.json()
    state["power"] = bool(d.get("power"))
    print(f"[admin ] power -> {state['power']}")
    return {"power": state["power"]}


@app.post("/admin/mode")
async def mode(req: Request):
    d = await req.json()
    state["mode"] = d.get("mode", state["mode"])
    return {"mode": state["mode"]}


def _stoptpu(k, d):
    if d.get("force"):
        ENGINES["tpu"]["force"] = True
    stop("tpu")
    ENGINES["tpu"].pop("force", None)


@app.post("/admin/action")
async def action(req: Request):
    d = await req.json()
    act = d.get("act", "")
    mapping = {"start-tpu": ("tpu", launch), "stop-tpu": ("tpu", lambda k: _stoptpu(k, d)),
               "start-gpu": ("gpu", launch), "stop-gpu": ("gpu", stop)}
    if act not in mapping:
        return JSONResponse({"error": "unknown act"}, status_code=400)
    key, fn = mapping[act]
    with lock:
        fn(key)
    return JSONResponse({act: "ok"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry", "live"], default="dry")
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()
    state["mode"] = args.mode
    threading.Thread(target=bg, daemon=True).start()
    print(f"controller mode={args.mode}  admin on 127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")