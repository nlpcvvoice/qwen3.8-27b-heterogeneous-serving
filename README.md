# Qwen3.8-27B — Heterogeneous Serving (TPU v5e + GPU T4)

Free high-end open-source LLM serving for agent/tool workloads: the same
**Qwen3.8-27B** model exposed as an OpenAI-compatible API from **two** free
accelerator backends, with automatic failover TPU → GPU.

```
                    clients (opencode, curl, agents...)
                     ┌──────────────┐   OpenAI-compatible /v1
                     │  controller +│   live: health-check, failover,
                     │    router    │   auto-registers endpoint/key
                     └──────┬───┬───┘
                    primary │   │ fallback
                            ▼   ▼
   TPU v5e-8 ─── vLLM (bf16, 262k ctx)      GPU 2xT4 ─── llama.cpp (Q4_K_M, llama-server)
   ~126.5 tok/s · queues 0-3h               ~13.6 tok/s · instant slots
```

Ready endpoints are persisted automatically to `tmp/current_services.json`
(`app/register_service.py take tpu|gpu` to force-register) — client machines
just read that one file for `endpoint` + `api_key`.

## Why / use case

- Self-host a flagship open-weight model at **\$0 inference cost** for
  evaluation, agent tool-calling and long-context workloads — versus per-token
  API pricing.
- The GPU leg acts as an on-demand **fallback** when the TPU queue/session is
  unavailable, so the service has a failover story, not a single point.
- Optional **paid tier** (Modal serverless H200, BF16, native 262k context) when
  the free legs are down or the workload needs ≥100k context — scale-to-zero,
  no idle billing; the free pair stays the default. See the 2026-09-11 benchmark.

## Model architecture & trade-offs

| | TPU v5e-8 (primary) | GPU 2xT4 (fallback) | H200 (paid, optional) |
|---|---|---|---|
| Tier | free primary | free fallback | paid single-card (serverless) |
| Precision | bf16 full | GGUF Q4_K_M (4-bit) | BF16 (no quant) |
| Engine | vLLM (XLA/TPU) | llama.cpp (CUDA sm_75) | vLLM 0.28 (stock) |
| Context | 262144 | 98304 total / 24576 per slot (KV q8_0) | 262144 native |
| Single-stream decode | 126.8 tok/s | 13.6 tok/s | 14.8 tok/s * |
| Concurrency | vLLM continuous batching | 4 parallel slots | max_num_seqs=8 batched |
| Long-context (≥100k) | batched, ~1.6k prefill tok/s | not fit (98k cap) | 4.5k prefill tok/s, 256k single |
| Cold start | ~25 min | ~31 min (v4) | ~7.6 min cold / ~1 min hot (snapshot) |
| Idle cost | free pool | free pool | $0 (min_containers=0) |
| Why | max quality*throughput | any-time availability | production long-context + full-BF16 quality |

\* single-stream on H200 is deliberately stock (no speculative decoding); see the
2026-09-11 caveat — it is a kernel-maturity artifact on this new architecture,
not a card-limit.

Key constraint discovered: the free "GPU T4 x2" slot is only obtained by
setting `machine_shape=NvidiaTeslaT4` in `kernel-metadata.json`; otherwise the
GPU pool silently falls back to P100 (pre-Ampere, sm_60). vLLM requires
Ampere+(sm≥8.0), so the T4/P100 leg must run llama.cpp.

## Model-level work

- **Quantization & VRAM budgeting**: Q4_K_M 16.5 GB weights fully offloaded to
  2xT4; KV cache sized to the VRAM ceiling (q8_0 → ~96k total ctx, verified
  against real runtime buffer prints).
- **Serving tuning**: `--parallel 4` slots + continuous batching; per-slot
  context isolation; batch 1024, 8 threads, CUDA arch 75.
- **Hardware/engine fit**: built llama.cpp from source inside the kernel image
  (the 2026 releases ship no Linux CUDA wheel); fixed the missing
  `libcuda.so.1` driver-library path/driver target discovery.
- **Custom evaluation**: reasoning trace capture, tool-call correctness probes
  (`list_files` etc.), single-stream + 4-way-concurrency self-tests on every
  deploy.

## Production concerns

| Concern | How handled |
|---|---|
| Model versioning | GGUF pinned from HF release; weights mirrored to a private Kaggle dataset (zero runtime download) |
| Observability | ntfy event bus + health checks on every startup phase; LLM observability (Langfuse/MLflow) planned |
| Latency | TTFT/tok-s tracked per run; benchmark gates each deploy |
| Accuracy drift | scheduled bf16 vs Q4 comparison + LLM-as-judge evals (planned) |
| Availability | TPU→GPU failover controller — **live**: unhealthy→failover, TPU healthy→GPU auto-recycled |
| Cost & quotas | separate TPU (~20h/wk) and GPU (~30h/wk) free quotas; keepalive-limited sessions. Paid tier (Modal H200): ~$4.54/h, `min_containers=0` + 600s scale-down + explicit stop → **$0 idle** (≈6.6 h/mo inside Modal's $30 free credit) |

## Measured results (2026-09-08)

| Test | TPU v5e-8 | GPU 2xT4 |
|---|---|---|
| Context window | 262,144 (vLLM bf16) | 98,304 (Q4_K_M + KV q8_0, 4 parallel slots) |
| Single-stream decode | 126.8 tok/s | 13.5 tok/s |
| 4 concurrent requests | (vLLM batching) | 4 req / 4.5 s, 0 errors |
| Tool-call emission | `list_files({"path":"."})` ✓ | — |
| Startup to ready | ~20 min (weight load + XLA compile) | ~2 min (weights offload) |
| Cold start (push → available) | ~25 min | ~31 min (measured v4, see below) |

Locust load test (5 VU, 70 s, mixed stream/sync, 15 realistic prompts — `loadtest/`, `app/locustfile.py`):

| Metric | TPU v5e-8 | GPU 2xT4 |
|---|---|---|
| RPS | 2.36 | 0.21 |
| TTFT p50 / p95 | 125 / 172 ms | 2.9 / 9.4 s |
| End-to-end p50 / p95 | 0.57 / 1.9 s | 16.6 / 37.7 s |
| Error rate | 0% | 0% |

Both engines stress-tested with zero failures; TPU ≈ 11-29x faster at low concurrency. Saturation ramp (20-50 VU) is next.

### Failover & service registration (verified 2026-09-09)

- Controller polls `healthy()` every loop; on TPU unhealthy → failover to GPU; on
  TPU healthy → primary auto-switched to TPU and GPU kernel is CONTROL-stopped.
- On READY (ready/serving/benchmark/heartbeat) the endpoint + api_key are
  atomically written to `tmp/current_services.json` — any client machine reads
  that file directly (guarded 600, never logged in plaintext).

### Cold-start latency (GPU 2xT4, measured 2026-09-08, kernel v4)

| Phase | Duration | Notes |
|---|---|---|
| push → kernel RUNNING | ~3 min | Kaggle queue + T4 slot provisioning |
| weights mount (private dataset) | <10 s | 16.46 GB already on Kaggle storage |
| llama.cpp source build (cmake, CUDA) | 25 min | i.e. 1511 s — this is the long pole |
| llama-server load + warmup | 2 min | model 15.3 GiB offload to 2xT4 VRAM |
| READY | — | 18:42:59, engine-built 18:40:52 |

**push → READY ≈ 31 min.** With the built `llama-server` cached in the private
dataset (roadmap item 5) the build phase disappears → **~4-6 min** cold start,
which is why the controller pre-warms the GPU only when needed.

## Three-accelerator benchmark (2026-09-11)

Same Qwen3.8-27B model served through three accelerators — one free TPU class,
one free GPU class, one paid single-card Hopper tier. All figures from live
endpoints in this repository; no external benchmarks imported.

```
                    clients (opencode, curl, agents...)
                     ┌──────────────┐   OpenAI-compatible /v1
                     │  controller +│
                     │    router    │
                     └───────┬──────┘
                 free pool   │   paid tier (Modal serverless)
                       ┌───────┴───────┐
                       │               │
        TPU v5e-8 ─── vLLM (bf16, 262k ctx, MTP4)    H200 ─── vLLM 0.28 (BF16, 262k ctx)
        ~126.8 tok/s single stream · 3.9 M tok KV     ~14.8 tok/s single · 5×457 tok in 36 s
        ~25 min push→READY (XLA + weights)             ~7.6 min cold / ~1 min hot (snapshot)
```

### Benchmark table

| Metric | TPU v5e-8 (free) | GPU 2×T4 (free) | H200 (paid, Modal) |
|---|---|---|---|
| Precision | bf16 + MTP4 | GGUF Q4_K_M (4-bit) | BF16 (no quant) |
| Engine | vLLM (XLA/TPU) | llama.cpp (CUDA sm_75) | vLLM 0.28, fp8 KV |
| Context window | 262,144 | 98,304 (KV q8_0, 4 slots) | 262,144 native (verified @256k) |
| Single-stream decode | 126.8 tok/s | 13.5 tok/s | 14.8 tok/s \* |
| TTFT (512-tok prompt) | ~0.13 s | ~2.9 s | 3.8 – 4.5 s |
| 4-way concurrent (64 tok each) | n/a (batched) | 4 req / 4.5 s, 0 err | 4 req / 5.4 s, 0 err |
| 5-stream wall (512 tok each) | n/a (batched) | n/a (4 slots max) | 36 s, 5/5 ok, 0 err |
| 5-stream per-stream tokens | — | — | 457 each |
| RPS (5 *concurrent* 512-tok streams, 100% succ) | — | — | 0.14 |
| RPS (Locust, short 15-prompt pool) | 2.36 | 0.21 | — |
| TTFT p50 / p95 (5-stream, 512-tok) | — | — | 6.2 / 14.4 s |
| E2E p50 / p95 (5-stream, 512-tok) | — | — | 13.5 / 36.0 s |
| Prefill, cold card — 7K / 28K / 113K prompt | 1,206 / 1,654 / 1,643 tok/s | — | 2,151 / 4,567 / 4,524 tok/s |
| 256k-token prompt TTFT / total | (batched) | — (max 98k) | 44.9 s / 73 s |
| Cold start (allocation → READY) | ~25 min | ~31 min | ~7.6 min cold / ~1 min hot (snapshot) |
| Error rate across every row | 0 % | 0 % | 0 % |

\* **Why the 14.8 tok/s single-stream is a software artifact, not a card limit.** The TPU leg
ships NVIDIA/Megatron MTP4 speculative decoding (each accepted step ≈4 tokens) plus
XLA-tuned kernels. The H200 path is deliberately stock vLLM 0.28 without MTP, and
Qwen3.8's hybrid linear-attention kernels are not yet optimized for Hopper in that
build — comparable dense models sustain roughly 60–100 tok/s on H200 once kernels
mature. Read single-stream as "stock-vLLM baseline", not "card ceiling".

**How to read the table.** H200's real wins are **long-context** and **quality**, not raw
throughput: it is the only leg here that serves the full 262,144 window on a single card
and sustains >4,000 tok/s prefill at ≥100k prompts (the TPU leg sits at ~1.6k and the T4
leg cannot fit >98k at all). Under the measured loads H200's aggregate decode is only
~1.2× the T4 leg (5 × 457 tok in 36 s ≈ 63 tok/s vs ~54 tok/s on 4 slots) — honest
number: at this output size the card's headroom needs more tokens per batch to show.
RPS must never be compared across output lengths here: TPU's 2.36 is a short-output
batch pool, H200's 0.14 is 512-token streams at 100% success.

**Recommendation matrix:** long-context / batched agent workloads ≥100k → H200 (paid,
recommended production tier for these); low-latency single-stream with speculative
decoding → TPU (free); instant availability → T4 leg (free failover).

### Why H200 (decisions record)

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| Accelerator tier | 1× H200 (Hopper, 141 GB HBM3, sm_90) | 4× A100 (VRAM split), 2× L40S (48 GB), 8× A100 (TP=8) | 262k at BF16 needs ~8 GiB KV + ~52 GiB weights on *one* device for simple ops; only one H200 has that headroom. A100×4 is cheaper $/token but needs TP orchestration and 80 GB/card caps; L40S×2 can't fit BF16 + 256k at once |
| Precision | BF16 (no quant) | FP8-W8A8, AWQ/GPTQ | the paid tier is *quality* — closes the gap between free bf16 (TPU) and Q4 (T4) |
| Engine | vLLM 0.28.0 (stock) | SGLang, TensorRT-LLM | only engine with a working native path for this hybrid linear-attention arch on Hopper; SGLang ships a separate runtime + different OpenAI shim |
| Speculative decoding | off | MTP4 | MTP4 on vLLM-Hopper = patchset + second head + tuning; not worth ops cost on a paid single card. Batched throughput + long-context work at stock |
| Cold start | vLLM SleepMode + GPU memory snapshot | full process boot | ~1 min hot vs ~7.6 min cold allocation; the cold figure is dominated by provider card allocation, weight load is ~1 min for the 52 GiB BF16 payload |
| Context sizing | 262144 native | 131072 (half) | 256k is the model's `max_position_embeddings`; fits comfortably at ~8 GiB fp8 KV |
| Availability | min_containers=0, scaledown_window=600 s, explicit stop | warm pool | no idle billing; ~1 min hot start via snapshot, ~7.6 min cold — fine for a paid tier used only when free legs are down |
| Endpoint | OpenAI-compatible, no auth | custom wire protocol | clients (opencode, LangChain, OpenAI SDK) work as-is; indistinguishable from the free legs |

**What this tier is not.** H200 here is a *quality and context* tier, not a throughput tier.
Higher single-stream throughput than TPU's 126.8 tok/s needs TPU×2 (TP=2) or a multi-H200
TP deployment — that is roadmap item 6, not the paid tier in this table.

## Scaling to an enterprise deployment

This repo runs on Kaggle's free quota (TPU ~20 h/wk, GPU ~30 h/wk, keepalive-limited
sessions). Every component maps 1:1 onto a paid, production-grade platform — the
skills exercised here transfer directly:

| Component (this repo) | Enterprise equivalent | Cloud/on-prem tooling |
|---|---|---|
| vLLM on TPU v5e | GPU-cluster vLLM / TensorRT-LLM / Triton | AWS SageMaker, KServe + NVIDIA Triton on EKS/GKE |
| llama.cpp Q4_K_M on 2xT4 | quantized inference at scale (FP8/AWQ/GPTQ), edge variants | ONNX Runtime, TensorRT, Triton, Jetson |
| Kaggle slot scheduling (`machine_shape`) | right-sized GPU instance pools, spot vs on-demand, autoscaling | K8s HPA/KEDA, SageMaker/Vertex pipelines, Argo/Airflow |
| ntfy events + local watcher | metrics, alerts, SLO/SLI, on-call | Prometheus + Grafana + Alertmanager, OTel → Datadog/New Relic |
| local fallback controller/router | gateway routing, canary + shadow, multi-region | LiteLLM proxy, Kong/Tyk, K8s service mesh |
| private Kaggle dataset (model mirror) | model registry + artifact versioning + rollout gates | MLflow, SageMaker / Vertex Model Registry, HF Hub |
| cloudflared tunnel + static key | mTLS / private link, WAF, rate limits, IAM/RBAC | API Gateway / ALB, Vault, OIDC |
| Locust self-run load tests | CI load pipeline, capacity & chaos testing | k6, Guidellm, LLMPerf, Inference Perf, Gremlin |
| LLM-as-judge + tool-call assertions | eval gates in CI/CD, drift & bias monitoring | DeepEval/promptfoo, Evidently, WhyLabs |

Skill transfer (what interviewers can probe):

| Skill built here | Enterprise application | Evidence in repo |
|---|---|---|
| Engine selection & trade-offs (bf16 vs Q4, HW constraints, KV budgeting) | choose deployment stack per cost/latency/quality budget | README tables + `kernel/serve_qwen38_gpu.py` |
| Quantization & VRAM budgeting (incl. KV-cache math) | FP8/AWQ/GPTQ choices on H100/A100; cache sizing in prod | budget formulas in this README |
| Heterogeneous scheduling (TPU + GPU via API) | capacity planning, instance right-sizing, failover design | push/probe scripts in `app/` |
| Failover & health routing | multi-region + gateway routing, SLO enforcement | controller design (roadmap) + watcher scripts |
| LLM observability planning | prod tracing, cost-per-token, eval dashboards | Langfuse/MLflow plan (roadmap) |
| Load testing & capacity analysis | capacity reports, saturation-point tuning | `loadtest/` + `app/locustfile.py` |

Business step: in a company the free-time constraints become an SLA
(99.9%+ availability, autoscaling to zero for dev, per-token cost accounting via
the observability layer, multi-tenant keys + quotas + guardrails).

## Reproduce

- Auth: `kaggle_login.py` (reads a local API-token file, in-memory only).
- Push the TPU kernel: `kaggle-tpu-lab/launch.py` (vLLM + cloudflared tunnel).
- Push the GPU kernel: `python app/push_gpu_serve.py` (llama.cpp, Q4_K_M).
- Watch live events: `python app/watch_*.py`.
- Read the live endpoint/key: `python app/register_service.py status` (or `take tpu|gpu`).

Secrets never enter this repository: tokens & runtime endpoint keys are kept in
`tmp/` state files and ignored via `.gitignore`.

## Skills Used

| Skill | Where |
|---|---|
| LLM serving (vLLM, llama.cpp, OpenAI-compat) | TPU + GPU kernels |
| Model optimization (GGUF Q4_K_M, offload, batching) | GPU engine |
| Heterogeneous accelerator scheduling (TPU/GPU via API, machine_shape) | push/probe scripts |
| Tool-call parsing & eval (qwen3_coder style) | TPU eval chain |
| Failover / health checking | controller — live on TPU+GPU (unhealthy→failover, healthy→recycle) | ✓ |
| Service endpoint registration | `tmp/current_services.json` + `register_service.py take/status` | ✓ |
| Load testing (Locust + TTFT/RPS/p95) | `loadtest/` — 5 VU run: TPU 0% error @ TTFT p95 172ms | ✓ |
| LLM observability (Langfuse/MLflow) | planned |
| LLM-as-judge evaluation | planned |
| Paid Hopper serving (H200, BF16, 262k, serverless) | H200 tier — vLLM 0.28 + SleepMode snapshot + scale-to-zero, Modal | ✓ |

## Roadmap

1. ~~TPU→GPU failover router~~ ✓ done (live controller, `app/controller.py`)
2. ~~Load testing~~ ✓ done (Locust, initial 5-VU baseline in `loadtest/`)
3. Saturation ramp 20-50 VU + LangGraph persona corpus
4. LLM observability (Langfuse/MLflow) + bf16-vs-Q4 eval
5. Cache prebuilt llama-server binary to skip the 25-min in-kernel build (pull `/kaggle/working` → private dataset)
6. H200 paid-tier throughput: raise `max_num_seqs`/batch for aggregate tok/s first; MTP4 on H200 as stretch (close the 14.8 → 126 tok/s single-stream gap); multi-H200 TP=2 for long-context scale

`kaggle-tpu-lab/` is derived from [kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab) (MIT).