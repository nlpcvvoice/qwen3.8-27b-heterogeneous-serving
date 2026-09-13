import os
import sys
import re
import json
import time
import glob
import shutil
import secrets
import tarfile
import collections
import threading
import subprocess
import urllib.request
from pathlib import Path

CFG = None  # __LAUNCHER_CONFIG__  (push_gpu_serve.py replaces this line)

DEFAULTS = {
    # 2xT4 (Kaggle "GPU T4 x2", machine_shape=NvidiaTeslaT4) quantized serving.
    "weights_dataset": "tentenshishi/qwen3-8-27b-q4-k-m-private",  # private mirror (16.46 GB)
    "hf_model_id": "unsloth/Qwen3.8-27B-GGUF",                    # fallback download source
    "hf_file": "Qwen3.8-27B-UD-Q4_K_M.gguf",
    "served_model_name": "qwen3.8-27b",
    "ctx_size": 98304,             # max with q8_0 KV on 2xT4 (~12.5 GiB cache, all weights on GPU)
    "n_parallel": 1,               # single stream: one slot gets the full 96K, not 96K/4
    "n_gpu_layers": 999,           # offload everything to the (two) T4s
    "n_batch": 1024,
    "n_threads": 8,
    "lcpp_extra_index": "https://abetlen.github.io/llama-cpp-python/whl/cu124",
    "keepalive_min": 420,
    "api_key": "",
    "ntfy_topic": "",
    "verbose": False,
}
CFG = {**DEFAULTS, **(CFG or {})}
_cfg_file = Path("serve_config.json")
if _cfg_file.exists():
    CFG.update(json.loads(_cfg_file.read_text()))
if not CFG["api_key"]:
    CFG["api_key"] = "sk-" + secrets.token_hex(16)

PORT = 8000
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("/tmp")
RAW_LOG = WORK / "llamacpp.log"
CLOUDFLARED = Path("/tmp/cloudflared")
BUILD = Path("/tmp/lcpp_build")
LLAMA_SERVER = BUILD / "bin" / "llama-server"
T0 = time.time()

os.environ["HF_HOME"] = "/tmp/hf"
_raw = open(RAW_LOG, "a", buffering=1)

# The GPU image keeps the driver userspace lib (libcuda.so.1) OUTSIDE the default
# ldconfig paths (NVIDIA container runtime layout: /usr/local/nvidia/lib64).
# cmake (CUDA::cuda_driver) and the llama.cpp CUDA backend both need it. Scan for
# it, put it on LD_LIBRARY_PATH, and symlink it into the toolkit dirs cmake searches.
for _d in ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib",
           "/usr/local/cuda/lib64", "/usr/local/cuda/targets/x86_64-linux/lib",
           "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu", "/usr/local/lib"):
    if os.path.isfile(os.path.join(_d, "libcuda.so.1")):
        _old = os.getenv("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = _d + (":" + _old if _old else "")
        print(f"[env] libcuda.so.1 found in {_d} -> LD_LIBRARY_PATH set", flush=True)
        _raw.write(f"[env] libcuda.so.1 found in {_d}\n")
        for _tgt in ("/usr/local/cuda/lib64", "/usr/local/cuda/targets/x86_64-linux/lib"):
            if not os.path.isdir(_tgt):
                continue
            for _lib in ("libcuda.so.1", "libcuda.so"):
                _src = os.path.join(_d, _lib)
                _dst = os.path.join(_tgt, _lib)
                if os.path.isfile(_src) and not os.path.exists(_dst):
                    try:
                        os.symlink(_src, _dst)
                    except Exception:
                        pass
        break


def log(*parts):
    line = time.strftime("[%H:%M:%S] ") + " ".join(str(p) for p in parts)
    print(line, flush=True)
    _raw.write(line + "\n")


def elapsed():
    return f"{int(time.time() - T0) // 60} min {int(time.time() - T0) % 60:02d} s"


def banner(step, title, note=""):
    log("")
    log("=" * 70)
    log(f" STEP {step}/6  {title}" + (f"   ({note})" if note else "") + f"   [{elapsed()} so far]")
    log("=" * 70)


def publish(phase, **extra):
    log(f"PHASE {phase}", json.dumps(extra) if extra else "")
    if not CFG["ntfy_topic"]:
        return
    try:
        body = {"topic": CFG["ntfy_topic"], "title": f"kaggle-tpu-lab-gpu {phase}",
                "message": json.dumps({"phase": phase, **extra})}
        req = urllib.request.Request("https://ntfy.sh", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        log(f"(ntfy publish failed: {e})")


def start_control_listener():
    """Remote kill switch: controller posts a CONTROL message to our ntfy topic;
    on 'stop' we shut down so the quota frees immediately."""
    topic = CFG.get("ntfy_topic")
    if not topic:
        return

    def _watch():
        since = int(time.time())
        while True:
            try:
                req = urllib.request.Request(
                    f"https://ntfy.sh/{topic}/json?poll=1&since={since}",
                    headers={"Accept": "text/event-stream"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    for raw in r:
                        line = raw.decode().strip()
                        if not line.startswith('{'):
                            continue
                        ev = json.loads(line)
                        since = max(since, ev.get("time", since))
                        if ev.get("event") != "message":
                            continue
                        msg = json.loads(ev.get("message", "{}"))
                        if msg.get("phase") == "CONTROL" and msg.get("cmd") == "stop":
                            log("CONTROL-STOP from controller — shutting down")
                            publish("auto-shutdown", reason="CONTROL-stop")
                            os._exit(0)
            except Exception:
                time.sleep(10)

    threading.Thread(target=_watch, daemon=True).start()


def sh(cmd, tag, show=None):
    show = CFG["verbose"] if show is None else show
    tail = collections.deque(maxlen=40)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        _raw.write(f"[{tag}] {line}\n")
        if show:
            print(f"[{tag}] {line[:400]}", flush=True)
    rc = p.wait()
    if rc != 0 and not show:
        log(f"[{tag}] exited with code {rc}; last lines:")
        for ln in list(tail)[-15:]:
            print("    " + ln[:300], flush=True)
    return rc


def find_input(*patterns):
    hits = []
    for pat in patterns:
        hits += glob.glob(f"/kaggle/input/{pat}") + glob.glob(f"/kaggle/input/datasets/*/{pat}")
    return (hits[0] if hits else None)


def fetch_cloudflared():
    if CLOUDFLARED.exists():
        return
    try:
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            CLOUDFLARED)
        CLOUDFLARED.chmod(0o755)
    except Exception as e:
        log(f"(cloudflared download failed: {e})")


threading = __import__("threading")

# ---------------- 1. weights ----------------
start_control_listener()
banner(1, "Model weights", "Q4_K_M ~16.5 GB GGUF")
threading.Thread(target=fetch_cloudflared, daemon=True).start()
weights_slug = CFG["weights_dataset"].split("/")[-1]
model_dir = find_input(weights_slug)
model_file = None
if model_dir:
    g = glob.glob(f"{model_dir}/**/*.gguf", recursive=True) + glob.glob(f"{model_dir}/*.gguf")
    if g:
        model_file = g[0]
if model_file and os.path.exists(model_file):
    publish("weights-mounted", path=str(model_file))
else:
    publish("weights-download", model=CFG["hf_model_id"], note="attach the GGUF dataset to skip this")
    hf_target = WORK / "model"
    hf_target.mkdir(exist_ok=True)
    from huggingface_hub import hf_hub_download
    model_file = hf_hub_download(CFG["hf_model_id"], CFG["hf_file"], local_dir=str(hf_target))
    publish("weights-downloaded")
log(f"   weights: {model_file} ({round(os.path.getsize(model_file) / 1e9, 2)} GB)")
_weights_t0 = time.time()

# ---------------- 2. engine ----------------
banner(2, "llama.cpp engine", "primary: cache-dataset llama-server; else source build (CUDA); else cu124 wheel")
t = time.time()
engine = None     # "llama-server" | "wheel"

# --- 2a. cache dataset hit -> skip the ~25 min source build -----------------
def package_engine(dst: Path) -> "Path | None":
    """Tar BUILD/bin + BUILD/lib into dst. None if no engine binary."""
    if not LLAMA_SERVER.exists():
        return None
    with tarfile.open(dst, "w:gz") as tf:
        for sub in ("bin", "lib"):
            d = BUILD / sub
            if d.is_dir():
                for f in sorted(d.iterdir()):
                    if f.is_file():
                        tf.add(f, arcname=f"{sub}/{f.name}")
    return dst


def unpack_engine(tar_path: Path) -> None:
    """Extract a cached engine.tar.gz back onto BUILD (same path -> RUNPATH ok)."""
    with tarfile.open(tar_path, "r:gz") as tf:
        try:
            tf.extractall(BUILD, filter="data")
        except TypeError:
            tf.extractall(BUILD)
    _srv = BUILD / "bin" / "llama-server"
    if _srv.exists():
        os.chmod(_srv, 0o755)


_cache_slug = (CFG.get("cache_dataset") or "").split("/")[-1]
_cached_tar = None
_cached_bin = None
if _cache_slug:
    _cached_tar = find_input(f"{_cache_slug}/engine.bundle")
    if not (_cached_tar and os.path.getsize(_cached_tar) >= 20_000_000):
        _cached_tar = None
    if not _cached_tar:
        _cached_tar = find_input(f"{_cache_slug}/engine.tar.gz")
    if _cached_tar and os.path.getsize(_cached_tar) < 20_000_000:  # sanity: >20 MB
        log(f"   cache tar too small, ignoring: {_cached_tar}")
        _cached_tar = None
    if not _cached_tar:
        _cached_bin = find_input(f"{_cache_slug}/llama-server")
        if _cached_bin and os.path.getsize(_cached_bin) < 1_000_000:  # sanity: >1 MB
            log(f"   cache binary too small, ignoring: {_cached_bin}")
            _cached_bin = None
if _cached_tar or _cached_bin:
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "bin").mkdir(parents=True, exist_ok=True)
    if _cached_tar:
        unpack_engine(_cached_tar)
        engine = "llama-server"
        publish("engine-cache-hit", secs=int(time.time() - t),
                src=os.path.basename(_cached_tar), size_mb=round(os.path.getsize(_cached_tar) / 1e6, 1),
                note="tarball (bin+lib): skipped the cmake build")
    else:
        shutil.copy2(_cached_bin, LLAMA_SERVER)
        os.chmod(LLAMA_SERVER, 0o755)
        engine = "llama-server"
        publish("engine-cache-hit", secs=int(time.time() - t),
                src=_cached_bin, size_mb=round(os.path.getsize(_cached_bin) / 1e6, 1),
                note="legacy single-file: skipped the cmake build")

have_toolchain = all(shutil.which(x) for x in ("cmake", "gcc", "make")) and (
    os.path.isfile("/usr/local/cuda/bin/nvcc") or shutil.which("nvcc") is not None)
if engine == "llama-server":
    pass  # cache hit, nothing more to do
elif have_toolchain:
    shutil.rmtree(BUILD, ignore_errors=True)
    build_cmd = ["bash", "-lc",
                 f"cmake -S /tmp/lcpp-src -B {BUILD} -DGGML_CUDA=ON "
                 f"-DGGML_CUDA_FORCE_DMMV=ON -DGGML_CCACHE=OFF "
                 f"-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 "
                 f"-DLLAMA_NATIVE=OFF && cmake --build {BUILD} --config Release -j "
                 f"$(nproc) --target llama-server"]
    # clone (shallow, pinned branch) then build
    if sh(["bash", "-lc", "git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/lcpp-src"],
          "git") == 0 and sh(build_cmd, "cmake") == 0 and LLAMA_SERVER.exists():
        engine = "llama-server"
        publish("engine-built", secs=int(time.time() - t), engine="llama-server (source, CUDA)")
        log(f"   built llama-server in {int(time.time() - t)} s")
        try:
            eng_tar = package_engine(WORK / "engine.tar.gz")
            if eng_tar is not None:
                publish("engine-cached", path=str(eng_tar),
                        size_mb=round(eng_tar.stat().st_size / 1e6, 1),
                        note="engine tarball (bin+lib) in /kaggle/working -> pullable when kernel completes")
            else:
                dst = WORK / "llama-server"
                shutil.copy2(LLAMA_SERVER, dst)
                publish("engine-cached", path=str(dst),
                        size_mb=round(dst.stat().st_size / 1e6, 1),
                        note="legacy single-file copy in /kaggle/working -> pullable when kernel completes")
        except Exception as e:
            log(f"(engine-cache package failed: {e})")
    else:
        finish = subprocess.run(["bash", "-lc",
                                 f"tail -20 {RAW_LOG} | grep -Ei 'cmake error|error:' | tail -4"],
                                capture_output=True, text=True).stdout.strip()
        publish("engine-build-failed", secs=int(time.time() - t), tail=finish[-500:] or "(no cmake error lines)")
        shutil.rmtree(BUILD, ignore_errors=True)
        log("   source build failed -> falling back to the cu124 wheel")
if not engine:
    pkgs = ["llama-cpp-python", "uvicorn", "fastapi", "sse-starlette"]
    if sh([sys.executable, "-m", "pip", "install", "-q", "--no-warn-script-location",
           "--extra-index-url", CFG["lcpp_extra_index"], *pkgs], "pip") == 0:
        engine = "wheel"
        ver = subprocess.run([sys.executable, "-c", "import llama_cpp; print(llama_cpp.__version__)"],
                             capture_output=True, text=True).stdout.strip()
        publish("engine-wheels", secs=int(time.time() - t), version=ver)
        log(f"   llama_cpp {ver} (cu124 wheel) ready in {int(time.time() - t)} s")
if not engine:
    publish("failed", step="engine")
    sys.exit(1)

# ---------------- 3. llama server ----------------
banner(3, "Starting llama server",
       f"engine={engine}, Q4_K_M on 2xT4, ctx {CFG['ctx_size']}, "
       f"parallel {CFG['n_parallel']} slots, gpu-layers {CFG['n_gpu_layers']}")


def server_args():
    if engine == "llama-server":
        return [str(LLAMA_SERVER), "-m", str(model_file),
                "--host", "127.0.0.1", "--port", str(PORT),
                "--alias", CFG["served_model_name"],
                "--api-key", CFG["api_key"],
                "-c", str(CFG["ctx_size"]),
                "-np", str(CFG["n_parallel"]),
                "-b", str(CFG["n_batch"]),
                "-t", str(CFG["n_threads"]),
                "-ngl", str(CFG["n_gpu_layers"]),
                "-ctk", "q8_0", "-ctv", "q8_0"]
    return [sys.executable, "-m", "llama_cpp.server",
            "--model", str(model_file),
            "--n_gpu_layers", str(CFG["n_gpu_layers"]),
            "--n_ctx", str(CFG["ctx_size"]),
            "--n_batch", str(CFG["n_batch"]),
            "--n_threads", str(CFG["n_threads"]),
            "--host", "127.0.0.1", "--port", str(PORT),
            "--alias", CFG["served_model_name"],
            "--api_key", CFG["api_key"]]


tail = collections.deque(maxlen=200)
if engine == "llama-server":
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD}/bin:{BUILD}/lib" + (f":{_ld}" if _ld else "")
p = subprocess.Popen(server_args(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def pump():
    for line in p.stdout:
        line = line.rstrip()
        if line:
            tail.append(line)
            _raw.write(f"[llamacpp] {line}\n")
            if CFG["verbose"]:
                print(f"[llamacpp] {line[:400]}", flush=True)


threading.Thread(target=pump, daemon=True).start()


def healthy():
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/models",
                                     headers={"Authorization": f"Bearer {CFG['api_key']}"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def wait_healthy(expect_min):
    t = time.time()
    while time.time() - t < 2700:
        if p.poll() is not None:
            tail_txt = "\n".join(list(tail)[-100:])
            log(f"server exited rc={p.returncode}; last output:\n{tail_txt}")
            publish("failed", step="server", rc=p.returncode, tail=tail_txt[-2500:])
            sys.exit(1)
        if healthy():
            return int(time.time() - t)
        el = int(time.time() - t)
        if el and el % 120 < 6:
            publish("compiling", elapsed_s=el)
            log(f"   ... {el // 60} min into startup (typically ~{expect_min} min)")
        time.sleep(5)
    publish("failed", step="health-timeout", tail="\n".join(list(tail)[-60:])[-2500:])
    sys.exit(1)


# ---------------- 4. tunnel ----------------
banner(4, "Public URL")
url = None
tunnel = None
for _ in range(60):
    if CLOUDFLARED.exists():
        break
    time.sleep(2)
if CLOUDFLARED.exists():
    tunnel = subprocess.Popen([str(CLOUDFLARED), "tunnel", "--url", f"http://127.0.0.1:{PORT}",
                               "--no-autoupdate", "--protocol", "quic"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    cf_lines = []

    def pump_cf():
        for line in tunnel.stdout:
            cf_lines.append(line.rstrip())
            _raw.write(f"[cloudflared] {line}")
    threading.Thread(target=pump_cf, daemon=True).start()
    deadline = time.time() + 180
    while time.time() < deadline and url is None:
        for ln in cf_lines:
            m = pat.search(ln)
            if m:
                url = m.group(0).rstrip("/")
                break
        time.sleep(1)
if url:
    log(f"   your endpoint will be  {url}/v1")
    log("   (not live yet — it answers 502 until the READY banner below)")
    publish("tunnel-url", endpoint=f"{url}/v1")
else:
    publish("tunnel-failed", note="server still reachable inside the kernel on :8000")

# ---------------- 5. wait, announce, self-test ----------------
startup = wait_healthy(12)
publish("serving", startup_secs=startup)
try:
    _vtxt = Path(RAW_LOG).read_text()
except Exception:
    _vtxt = ""
_vr = [ln.strip() for ln in _vtxt.splitlines()
       if re.search(r"buffer size =|VRAM used|model size =|KV self size", ln)]
_smi = []
try:
    _smi_out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20)
    _smi = [ln.strip() for ln in _smi_out.stdout.splitlines() if ln.strip()]
except Exception:
    _smi = []
if _vr or _smi:
    publish("vram", ctx_size=CFG["ctx_size"], n_parallel=CFG["n_parallel"],
            kv="q8_0", n_lines=len(_vr), lines="\n".join(_vr[-20:])[-1200:],
            n_smi=len(_smi), smi=_smi)
log("")
log("#" * 70)
log(f"#  READY — the server is live ({elapsed()} after start)")
log(f"#  ENDPOINT : {url + '/v1' if url else 'http://127.0.0.1:8000/v1 (tunnel failed)'}")
log(f"#  API KEY  : {CFG['api_key']}")
log(f"#  MODEL    : {CFG['served_model_name']}  (ctx {CFG['ctx_size']}, "
    f"{CFG['n_parallel']} parallel slots, 2xT4)")
log("#" * 70)
publish("ready", endpoint=(f"{url}/v1" if url else None), api_key=CFG["api_key"],
        model=CFG["served_model_name"], ctx_size=CFG["ctx_size"],
        n_parallel=CFG["n_parallel"], engine=engine, startup_secs=startup,
        weights_gb=round(os.path.getsize(model_file) / 1e9, 2))


def completion(text, max_tokens, stream=False, timeout=900):
    body = {"model": CFG["served_model_name"], "prompt": text,
            "max_tokens": max_tokens, "temperature": 0.0}
    if stream:
        body.update(stream=True)
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {CFG['api_key']}"})
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    t0 = time.time(); ttft = None; gen = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            ch = obj.get("choices") or []
            if ch and ch[0].get("text"):
                if ttft is None:
                    ttft = time.time() - t0
                gen += 1
    return ttft, time.time() - t0, gen


def chat(messages, max_tokens=32, timeout=600):
    body = {"model": CFG["served_model_name"], "messages": messages,
            "max_tokens": max_tokens, "temperature": 0.0}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {CFG['api_key']}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def self_test():
    tps = None
    try:
        txt = completion("The capital of France is", 8)["choices"][0]["text"]
        ttft, total, gen = completion("Write a short story about a lighthouse.", 192, stream=True)
        tps = (gen - 1) / (total - ttft) if gen > 1 else 0.0
        publish("benchmark", decode_tok_s=round(tps, 1), sanity=txt.strip()[:60])
        log(f"   self-test: {tps:.1f} tok/s single-stream decode; "
            f"'The capital of France is' -> {txt.strip()[:40]!r}")
    except Exception as e:
        publish("benchmark-error", err=str(e)[:200])
        log(f"   SELF-TEST FAILED: {e}")

    # the user question answered empirically: 4 simultaneous requests
    n = 4
    errs = []
    t0 = time.time()
    def one():
        try:
            completion("Count from one to twenty in words.", 24, timeout=300)
        except Exception as e:
            errs.append(str(e)[:120])
    ths = [threading.Thread(target=one) for _ in range(n)]
    [x.start() for x in ths]
    [x.join() for x in ths]
    wall = time.time() - t0
    publish("parallel4-test", requests=n, wall_secs=round(wall, 1), errors=errs)
    log(f"   {n} simultaneous requests finished in {wall:.1f}s "
        f"({'ok' if not errs else 'ERR ' + str(errs)})")
    return tps


banner(6, "Self-test", "single-stream speed + 4-way concurrency; usable meanwhile")
self_test()

t_serve = time.time()
while time.time() - t_serve < CFG["keepalive_min"] * 60:
    time.sleep(120)
    if p.poll() is not None:
        publish("stopped", reason="server-exit", rc=p.returncode)
        sys.exit(1)
    up = int((time.time() - t_serve) / 60)
    if up % 10 < 2:
        publish("heartbeat", up_min=up, endpoint=(f"{url}/v1" if url else None))
        log(f"   still serving ({up} min) — {url + '/v1' if url else ''}")
publish("auto-shutdown", served_min=CFG["keepalive_min"])
p.terminate()
sys.exit(0)