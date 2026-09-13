# t4-kit — self-contained Kaggle T4 job package

> Lets another agent in any other project directory authenticate to Kaggle, push a
> Qwen3.8-27B inference job onto 2xT4, control it in real time and fetch results —
> all from this folder alone. **GPU T4 only, no TPU logic.**
>
> Engine capability is identical to the main repo's GPU kernel: llama.cpp CUDA
> source build / engine cache (skips the 25-min build) / OpenAI-compatible serving /
> event monitoring / quota control / output download / self-test benchmark.

| Item | Value |
|---|---|
| Hardware | Kaggle GPU T4 x2 (`NvidiaTeslaT4`, 16 GB each), quantized Q4_K_M |
| Model | Qwen3.8-27B (GGUF `UD-Q4_K_M`) |
| Engine | llama.cpp `llama-server` (OpenAI-compatible `/v1`) |
| Context | ctx 98304 · q8_0 KV · parallel 4 slots · full GPU offload |
| Weekly quota | GPU 30 h (queue waits automatically on overage; TPU not used) |
| Control channel | ntfy topic (streaming events + CONTROL-stop remote shutdown) |

---

## 1. Layout

| File | Role |
|---|---|
| `README.md` | this doc — the only thing you need to read |
| `start.sh` | main entrypoint (`test/push/watch/stop/download/cache-*`) |
| `kit_config.py` | constants: account/datasets/paths, one-time configuration |
| `kaggle_login.py` | auth: in-memory token, silent banner, quota query (GPU only) |
| `push_job.py` | push: inject CFG + attach datasets → push T4 kernel |
| `kernel/serve_t4_gpu.py` | kernel workload (6-step pipeline) |
| `watch_job.py` | monitoring: poll events + kernel status; optional auto-stop after build |
| `stop_job.py` | stop: send CONTROL-stop, release quota immediately |
| `download_output.py` | download: kernel output → `<project>/out/<version>/` |
| `engine_cache.py` | cache: validate / upload / fetch `engine.tar.gz` |
| `check.py` | diagnostics: token/quota/kernel/dataset status |

Runtime state (never committed): `<project>/tmp/t4-kit/` (job state, kaggle_cfg, logs, staging).

---

## 2. Prerequisites

| Item | Requirement | Notes |
|---|---|---|
| Python | ≥3.10 | use the project's Venv if present |
| kaggle SDK | ≥2.1 | `pip install kaggle` (inside Venv) |
| Token | `reference/API-Token` | lookup order: env `KAGGLE_TOKEN_FILE` → `reference/API-Token` here → `<project root>/reference/API-Token` |
| Dataset | private GGUF mirror | 16.46 GB; if missing the kernel falls back to HF download |
| Egress | `api.kaggle.com` + `ntfy.sh` reachable | push/query + events/control |
| Quota | GPU >0.5 h | `./start.sh check` |

---

## 3. Deploy into a new project (once)

| Step | Action | Output |
|---|---|---|
| 1 | `cp -r t4-kit <proj>/` | carry all deps |
| 2 | place token at `<proj>/reference/API-Token` (or `<proj>/t4-kit/reference/API-Token`) | in-memory read |
| 3 | install kaggle inside the project Venv (or `python3`) | SDK ready |
| 4 | `./start.sh test` | offline green == deployable |
| 5 | `./start.sh check` | token/quota/datasets OK |

> Multiple projects share nothing: outputs go to `<proj>/out/`, state to `<proj>/tmp/t4-kit/`.

---

## 4. CLI reference (the 8 commands)

| Command | Action | Typical args |
|---|---|---|
| `./start.sh test` | offline self-test (0 quota) | — |
| `./start.sh check` | token/quota (kernel/dataset) diagnostics | `--quiet` |
| `./start.sh push` | push T4 kernel | `--keepalive-min 120` · `--no-cache` · `--dry` |
| `./start.sh watch` | event monitor | `--stop-on built` (auto-stop after compile) / `serve` (default, keepalive-bound) · `--timeout-min` |
| `./start.sh stop` | stop kernel now (release GPU) | — |
| `./start.sh download` | download outputs | `--out DIR` |
| `./start.sh cache-upload <tar>` | upload engine cache (private dataset) | `--sha` · `--msg` |
| `./start.sh cache-fetch` | pull + verify engine cache | `--check` |

> `./start.sh push --help` / `watch --help` list all options.

---

## 5. Job lifecycle

```
push ──▶ queued ──▶ 1 weights ──▶ 2 engine ──▶ 3 llama-server ──▶ 4 tunnel ──▶ 5 READY ──▶ 6 self-test ──▶ keepalive ──▶ stop
        │                 │   │                                   │                    ├─ single-stream tok/s bench
        │                 │   └─ cache-hit (seconds)              │                    └─ 4-way concurrency
        │                 └─ source build (~1452s) + package engine.tar.gz
```

**Automatic in-kernel decision:**

| Engine source | Trigger | Time |
|---|---|---|
| A cache-hit | run mounts `llama-server-qwen38-cache` (with `engine.tar.gz`) | seconds |
| B source build | cmake + CUDA build (sm_75), packaged to `/kaggle/working` | ~25 min |
| C cu124 wheel | pip install llama-cpp-python when A/B unavailable | ~2-5 min |

---

## 6. Event stream (ntfy topic, readable live via `watch`)

| phase | meaning | key fields |
|---|---|---|
| `weights-mounted` / `weights-downloaded` | GGUF ready | `path` |
| `engine-cache-hit` | cache hit, build skipped | `secs`,`size_mb` |
| `engine-built` | source build succeeded | `secs` |
| `engine-cached` | engine.tar.gz on disk (download-ready) | `path`,`size_mb` |
| `engine-build-failed` | build failed (falls back to wheel) | `tail` |
| `tunnel-url` | Cloudflare public URL | `endpoint` |
| `serving` / `ready` | health ok / OpenAI API up | `startup_secs`,`endpoint` |
| `benchmark` / `parallel4-test` | self-test metrics | `decode_tok_s`,`wall_secs` |
| `heartbeat` | every 10 min | `up_min` |
| `auto-shutdown` / `stopped` | end reason | `reason` / `served_min` |

---

## 7. Engine cache: build once, reuse forever (the cost saver)

```
round 1 (no cache):  push ──▶ watch --stop-on built ──▶ engine-cached ──▶ auto CONTROL-stop
                               │                       │
                               ▼                       ▼
                        download(engine.tar.gz)    cache-upload → private dataset
round 2+:            push(auto-attaches cache) ──▶ cache-hit, seconds to ready
```

| Rule | Value |
|---|---|
| Cached content | `bin/llama-server` + `bin/libllama-server-impl.so` + `lib/*.so` (thin 17 KB shell is not usable alone) |
| Pre-upload check | `cache-validate`: gzip tar containing launcher + runtime libs |
| Fetch check | `cache-fetch`: sha256 vs `manifest.json` + structure re-check |
| Visibility | private dataset; push is free (no GPU quota) |
| Repair dirty cache | `push --no-cache` (force source build) → re-upload to overwrite |

---

## 8. Quota & cost control

| Item | Advice | Why |
|---|---|---|
| `--keepalive-min` | 120 (default) | service upper bound; kernel self-stops at expiry |
| after cache hit | `watch --stop-on built` auto-stops | zero idle burn |
| overage | kaggle queues automatically, no cost, just wait | 30 h/week reset visible via `check` |
| donated credits | free; paid-credit machines only if chosen | kit is pinned to `NvidiaTeslaT4` (free tier) |

---

## 9. FAQ

| Symptom | Cause | Fix |
|---|---|---|
| `No token found` | token not on a known path | see Prerequisites; or set `KAGGLE_TOKEN_FILE` |
| `push` says cache dataset not found | engine cache never uploaded | normal: first round uses source build |
| `watch` reports `ntfy poll error` | local→ntfy.sh disconnected | retry; restart `watch`; re-send stop in window |
| kernel RUNNING but not READY | engine compiling (~25 min) | watch for `engine-built` |
| `engine-build-failed` | environment/network issue | push falls back to cu124 wheel; or retry `--no-cache` |
| must stop immediately | over budget | `./start.sh stop` (kernel `os._exit` on CONTROL, quota released now) |
| kernel COMPLETE, watchdog not stopped manually | keepalive expiry | set `watch --timeout-min` ≥ keepalive |

---

## 10. Security invariants

| Forbidden | Requirement |
|---|---|
| print/upload the token | show mask + length only; token in memory |
| write `~/.kaggle` | everything under `<project>/tmp/` |
| commit runtime state | `tmp/` is gitignored |
| introduce TPU concepts | no TPU anywhere in this kit's kernel/scripts/docs |

---

## 11. Minimal workflow (quick reference)

| Scenario | Command sequence |
|---|---|
| first full flow | `test` → `check` → `push` → `watch --stop-on built` → `download` → `cache-upload <out>/<v>/engine.tar.gz` → done |
| reuse cache for a job | `push` → `watch --timeout-min 300` → call the model via the `ready` endpoint → keepalive self-stop or manual `stop` |
| fetch results only | `download` (`out/<version>/` holds kernel artifacts) |