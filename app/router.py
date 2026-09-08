#!/usr/bin/env python3
"""Local router/controller for the TPU+GPU heterogeneous serving (P1).

Single stable OpenAI-compatible endpoint that auto-fails over TPU -> GPU and
exposes pinned ids for explicit engine selection:
  - qwen3.8-27b       auto   (healthy TPU first, GPU fallback)
  - qwen3.8-27b-tpu   pinned TPU
  - qwen3.8-27b-gpu   pinned GPU

Behaviour:
  - periodic upstream health checks (/v1/models, 30 s)
  - SSE streaming preserved, non-stream passed through
  - upstream model id rewritten to the engine's real id
  - 'auto': one retry on the other engine on connection-level failure
  - /healthz exposes engine liveness (for scheduler/eco-mode wiring)

Config: tmp/router_config.json (default), overridable via ROUTER_CONFIG env.
Run:   ./Venv/bin/python app/router.py
"""
import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = Path(os.environ.get("ROUTER_CONFIG", ROOT / "tmp/router_config.json"))

app = FastAPI(title="qwen3.8-27b router")

cfg = None
client = None
health = {"tpu": False, "gpu": False}
last_up = {}   # model id -> upstream key actually used (telemetry)

CTL_STATE = ROOT / "tmp" / "controller_state.json"


def _primary():
    """Scheduler-preferred engine for 'auto'; default tpu."""
    try:
        return (json.loads(CTL_STATE.read_text()) or {}).get("primary") or "tpu"
    except Exception:
        return "tpu"


def load_config():
    return json.loads(CONFIG_FILE.read_text())


def _other(key):
    return "gpu" if key == "tpu" else "tpu"


def _up(key):
    return cfg["upstreams"][key]


def pick(rid):
    if rid == cfg["ids"]["auto"]:
        pref = _primary()
        return pref if health.get(pref) else (_other(pref) if health.get(_other(pref)) else None)
    if rid == cfg["ids"]["tpu"]:
        return "tpu" if health["tpu"] else None
    if rid == cfg["ids"]["gpu"]:
        return "gpu" if health["gpu"] else None
    return None


async def check_upstream(key):
    up = _up(key)
    try:
        r = await client.get(f"{up['endpoint']}/models",
                             headers={"Authorization": f"Bearer {up['api_key']}"},
                             timeout=8)
        health[key] = r.status_code == 200 and up["model"] in r.text
    except Exception:
        health[key] = False
    return health[key]


async def health_loop():
    while True:
        await asyncio.gather(check_upstream("tpu"), check_upstream("gpu"))
        await asyncio.sleep(30)


@app.on_event("startup")
async def startup():
    global cfg, client
    cfg = load_config()
    client = httpx.AsyncClient(timeout=httpx.Timeout(1200.0, connect=10.0))
    asyncio.create_task(health_loop())


async def _stream(resp):
    try:
        async for chunk in resp.aiter_bytes():
            yield chunk
    finally:
        await resp.aclose()


async def _forward(key, body, path, query):
    up = _up(key)
    url = f"{up['endpoint']}/{path}" + (f"?{query}" if query else "")
    headers = {"Authorization": f"Bearer {up['api_key']}",
               "Content-Type": "application/json"}
    body["model"] = up["model"]
    try:
        req = client.build_request("POST", url, json=body, headers=headers)
        resp = await client.send(req, stream=True)
    except Exception as e:              # connection-level failure -> retryable
        return None, str(e)
    if body.get("stream"):
        return StreamingResponse(_stream(resp), media_type="text/event-stream"), None
    txt = await resp.aread()
    await resp.aclose()
    ct = resp.headers.get("content-type", "application/json")
    return httpx.Response(resp.status_code, content=txt,
                          headers={"content-type": ct}), None


@app.get("/healthz")
async def healthz():
    return {**health, "last_up": last_up, "ts": time.time()}


@app.get("/v1/models")
async def models():
    data = [{"id": n, "object": "model", "created": int(time.time()),
             "owned_by": "router",
             "routing": "auto" if n == cfg["ids"]["auto"] else n.split("-")[-1],
             "engines": {"tpu": health["tpu"], "gpu": health["gpu"]}}
            for n in cfg["ids"].values()]
    return JSONResponse({"object": "list", "data": data})


@app.post("/v1/{path:path}")
async def chat(path: str, request: Request):
    body = await request.json()
    rid = body.get("model", cfg["ids"]["auto"])
    key = pick(rid)
    if key is None:
        return JSONResponse({"error": {"message": "no healthy upstream",
                                       "type": "no_upstream", "model": rid}},
                            status_code=503)
    resp, err = await _forward(key, body, path, request.url.query)
    if err is not None and rid == cfg["ids"]["auto"]:
        alt = "gpu" if key == "tpu" else "tpu"
        if health[alt]:
            resp, err = await _forward(alt, body, path, request.url.query)
            key = alt if err is None else key
    if err is not None:
        return JSONResponse({"error": {"message": err, "type": "upstream_error"}},
                            status_code=502)
    last_up[rid] = key
    if isinstance(resp, StreamingResponse):
        return resp
    return JSONResponse(content=json.loads(resp.content), status_code=resp.status_code)


if __name__ == "__main__":
    import uvicorn

    c = load_config()
    uvicorn.run(app, host=c.get("listen", {}).get("host", "127.0.0.1"),
                port=c.get("listen", {}).get("port", 8080), log_level="info")