# Qwen3.8-27B — Heterogeneous Serving (TPU v5e + GPU T4)

*LLM inference serving with vLLM · llama.cpp · quantization · automatic failover · load testing · containerized build*

Free high-end open-source LLM serving for agent/tool workloads: the same
**Qwen3.8-27B** model exposed as an OpenAI-compatible API from **two** free
accelerator backends, with automatic failover TPU → GPU.

A controller/router picks a healthy backend and exposes one OpenAI-compatible
endpoint + key that client machines read from a single state file.

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
  no idle billing; the free pair stays the default. See the 2026-09-14 re-measured
  benchmark (H200 now runs SGLang + MTP NEXTN k=3 speculative decoding).

## Model architecture & trade-offs

| | TPU v5e-8 (primary) | GPU 2xT4 (fallback) | H200 (paid, optional) |
|---|---|---|---|
| Tier | free primary | free fallback | paid single-card (serverless) |
| Precision | bf16 full | GGUF Q4_K_M (4-bit) | BF16 (no quant) |
| Engine | vLLM (XLA/TPU) | llama.cpp (CUDA sm_75) | SGLang (MTP NEXTN k=3, CUDA graphs) |
| Context | 262144 | 98304 total / 24576 per slot (KV q8_0) | 262,144 native (verified to 254,987) |
| Single-stream decode | 126.8 tok/s | 13.6 tok/s | 44.2 tok/s (MTP) * |
| Concurrency | vLLM continuous batching | 4 parallel slots | max-running-requests 8 (SGLang) |
| Long-context (≥100k) | batched, ~1.6k prefill tok/s | not fit (98k cap) | 254,987-tok prompt served (TTFT 30.8 s); prefill ~9.4k tok/s @113K |
| Cold start | ~25 min | ~31 min (v4) | 2.6–3.1 min (156–186 s, no snapshot) |
| Idle cost | free pool | free pool | $0 (min_containers=0) |
| Why | max quality*throughput | any-time availability | production long-context + full-BF16 quality |

\* H200 single-stream under SGLang + MTP NEXTN k=3 (the model's native, in-checkpoint
draft head — no patch) with CUDA graphs, BF16. A single H200 streaming ~52 GiB of
weights once per token is memory-bandwidth-bound at ≈88 tok/s (~4.8 TB/s HBM3), so
44.2 tok/s ≈ 50 % of that ceiling on the SGLang path — speculative decoding already
closes most of the 14.8 tok/s stock-vLLM gap.

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
| Availability | TPU→GPU failover controller (local process, E2E-verified 2026-09-09): unhealthy→failover, TPU healthy→GPU auto-recycled |
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

Both engines held 0% errors under the measured 5-VU load; TPU ≈ 11-29x faster at
this concurrency level. Saturation ramp (20-50 VU) is next.

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
| llama.cpp source build (cmake, CUDA) | 25 min | i.e. 1511 s — the dominant phase |
| llama-server load + warmup | 2 min | model 15.3 GiB offload to 2xT4 VRAM |
| READY | — | 18:42:59, engine-built 18:40:52 |

**push → READY ≈ 31 min.** With the built `llama-server` cached in the private
dataset (roadmap item 5) the build phase disappears → **~4-6 min** cold start,
which is why the controller pre-warms the GPU only when needed.

### Engine factory — containerized offline build (no GPU)

The ~25 min in-kernel cmake build can be moved off Kaggle: a Docker image
(`nvidia/cuda:12.4.0-devel`, nvcc on CPU only) produces the *identical*
`engine.tar.gz` (`bin/` + `lib/`, same cmake flags as the in-kernel source
build). It is uploaded to the private dataset and flows through the existing
`bootstrap_cache.py` / `unpack_engine()` pipeline, so kernels boot with
`engine-cache-hit` (≈instant) even on a first run.

| | In-kernel source build | engine-factory (Docker) |
|---|---|---|
| Where | inside the Kaggle kernel (GPU session, card idle) | any CPU machine / CI |
| Flags | GGML_CUDA=ON / sm_75 / FORCE_DMMV / NATIVE=OFF | identical (see Dockerfile) |
| Result | `WORK/engine.tar.gz` | `/out/engine.tar.gz` + pinned `engine.sha` |
| Cost | ~25 min GPU-session + Kaggle resources | local CPU; zero Kaggle quota |
| Reproducible | depends on build-day deps | same image + commit → identical artifact |

Usage: `kaggle-tpu-lab/engine-factory/build_engine.sh [OUT] [--commit <sha|master>]`
(details in `engine-factory/README.md`). Records an `engine.sha` for version
pinning; a T4 runtime smoke-test is the only remaining in-kernel check.

## Three-accelerator benchmark (2026-09-11; H200 re-measured 2026-09-14)

Same Qwen3.8-27B model served through three accelerators — one free TPU class,
one free GPU class, one paid single-card Hopper tier. All figures from live
endpoints in this repository; no external benchmarks imported.

Client-visible topology for the three-leg setup:

```
                    clients (opencode, curl, agents...)
                     ┌──────────────┐   OpenAI-compatible /v1
                     │  controller +│
                     │    router    │
                     └───────┬──────┘
                 free pool   │   paid tier (Modal serverless)
                       ┌───────┴───────┐
                       │               │
         TPU v5e-8 ─── vLLM (bf16, 262k ctx, MTP4)       H200 ─── SGLang (BF16, 262k ctx, MTP k=3)
         ~126.8 tok/s single · 3.9 M tok KV              44.2 single · 200.4 tok/s @8-way concurrent
         ~25 min push→READY (XLA + weights)              prefill ~9.4k tok/s @113K · 2.6–3.1 min cold
```

### Benchmark table

| Metric | TPU v5e-8 (free) | GPU 2×T4 (free) | H200 (paid, Modal) |
|---|---|---|---|
| Precision | bf16 + MTP4 | GGUF Q4_K_M (4-bit) | BF16 (no quant) |
| Engine | vLLM (XLA/TPU) | llama.cpp (CUDA sm_75) | SGLang (MTP NEXTN k=3, CUDA graphs) |
| Context window | 262,144 | 98,304 (KV q8_0, 4 slots) | 262,144 native (configured) |
| Single-stream decode | 126.8 tok/s | 13.5 tok/s | 44.2 tok/s \* (median 39.8–44.5) |
| TTFT (512-tok prompt) | ~0.13 s | ~2.9 s | 0.35 s |
| 4-way concurrent (64 tok each) | n/a (batched) | 4 req / 4.5 s, 0 err | 144.5 @2-way / 200.4 @8-way (128-tok burst; 4-way run 99.8 anomalous, ‡) |
| 5-stream wall (512 tok each) | n/a (batched) | n/a (4 slots max) | 4.77 s (5 × 561-tok, 5/5 ok) |
| 5-stream per-stream tokens | — | — | 561 each |
| RPS (5 *concurrent* 512-tok streams, 100% succ) | — | — | 1.05 (5 × 561-tok) |
| RPS (Locust, short 15-prompt pool) | 2.36 | 0.21 | 2.03 |
| TTFT p50 / p95 (5-stream, 512-tok) | — | — | 0.352 / 0.417 s (Locust 5 VU) |
| E2E p50 / p95 (5-stream, 512-tok) | — | — | 0.76 / 2.41 s (Locust 5 VU) |
| Prefill, cold card — 7K / 28K / 113K prompt | 1,206 / 1,654 / 1,643 tok/s | — | 7,831 / 9,017 / 9,395 tok/s |
| 256k-token prompt TTFT / total | (batched) | — (max 98k) | 30.8 s / 42.0 s (254,987 / 242,570 tok) |
| Cold start (allocation → READY) | ~25 min | ~31 min | 2.6–3.1 min (156–186 s, no snapshot) |
| Error rate across every row | 0 % | 0 % | 0 % |

\* **Why H200 single-stream is 44.2 tok/s, and where the ceiling sits.** The TPU leg
ships NVIDIA/Megatron MTP4 speculative decoding (each accepted step ≈4 tokens). The
H200 path now runs SGLang with **MTP NEXTN k=3** — the model's native, in-checkpoint
draft head (no patch) — plus CUDA graphs. A single H200 streaming the ~52 GiB BF16
weights once per token is memory-bandwidth-bound at ≈88 tok/s (~4.8 TB/s HBM3), so
44.2 tok/s ≈ 50 % of that ceiling on the SGLang path. Read the figure as
"SGLang + MTP baseline", not "card ceiling". Sample count/method and the
MTP-vs-greedy lossless A/B gate are part of the correctness verification (pending).

† H200 rows formerly marked "—" were re-measured on 2026-10-30 under SGLang 0.5.11
+ MTP NEXTN k=3 (isolated account, billed \$0.00). The vLLM-era figures they
supersede (prefill 2,151 / 4,567 / 4,524 tok/s; 256k prompt TTFT/total 44.9 s / 73 s;
5-stream 36 s at 457 tok each; TTFT p50/p95 6.2/14.4 s) are retired. MTP draft
acceptance in these runs: 53.3–54.2 % (≈2.60–2.63 tokens per decode step).

‡ The 4-way concurrent run (128-tok burst, 2026-10-30) measured 99.8 tok/s
aggregate — below the 2-way (144.5) and 8-way (200.4) values, i.e. non-monotonic
in concurrency. It was flagged anomalous in the source bench data and is excluded
from the headline figures; the 2-way/8-way values are used for the scaling claim.

**How to read the table.** H200's real wins are **long-context**, **concurrent
throughput** and **quality**: it is the only leg here configured for the full 262,144
window on a single card (the TPU leg also fits 262k natively; the T4 leg cannot fit
>98k at all), it scales ~4.5× from 44.2 single-stream to 200.4 tok/s at 8-way
concurrency (continuous batching + MTP), and it pays the single-stream gap vs the
TPU leg (126.8 tok/s) for stability, full-BF16 quality and no free-quota cap
(decisions record below). RPS must never be compared across output lengths here:
TPU's 2.36 and H200's 2.03 are short-output Locust pools; H200's 1.05 is the
5-concurrent 561-tok stream rate.

**Recommendation matrix:** long-context / batched agent workloads ≥100k → H200 (paid,
recommended production tier for these); low-latency single-stream with speculative
decoding → TPU (free); instant availability → T4 leg (free failover).

### Why H200 (decisions record)

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| Accelerator tier | 1× H200 (Hopper, 141 GB HBM3, sm_90) | 4× A100 (VRAM split), 2× L40S (48 GB), 8× A100 (TP=8) | 262k at BF16 needs ~8 GiB KV + ~52 GiB weights on *one* device for simple ops; only one H200 has that headroom. A100×4 is cheaper $/token but needs TP orchestration and 80 GB/card caps; L40S×2 can't fit BF16 + 256k at once |
| Precision | BF16 (no quant) | FP8-W8A8, AWQ/GPTQ | the paid tier is *quality* — closes the gap between free bf16 (TPU) and Q4 (T4) |
| Engine | SGLang (MTP NEXTN k=3, CUDA graphs) | vLLM 0.28 (stock), TensorRT-LLM | SGLang matured a native MTP NEXTN + CUDA-graphs path for this hybrid linear-attention arch on Hopper → single-stream 14.8 → 44.2 tok/s with the in-checkpoint MTP head, no patchset; its OpenAI-compatible shim keeps all clients unchanged (what changed, 2026-09-14) |
| Speculative decoding | MTP NEXTN k=3 (in-checkpoint) | off (stock vLLM) | in-checkpoint MTP head needs no patch; ~3× single-stream under SGLang (14.8 → 44.2 tok/s). Lossless-vs-greedy A/B gate is part of the kernel correctness verification |
| Cold start | cold boot; triton/CUDA kernels pre-baked in image | vLLM SleepMode + GPU memory snapshot | the snapshot path proved fragile/failing in practice; pre-baked kernels reach READY in 2.6–3.1 min (156–186 s) with no snapshot state to manage |
| Context sizing | 262144 native | 131072 (half) | 256k is the model's `max_position_embeddings`; fits comfortably at ~8 GiB fp8 KV |
| Availability | min_containers=0, scaledown_window=600 s, explicit stop | warm pool | no idle billing; 2.6–3.1 min cold start to READY — fine for a paid tier used only when free legs are down |
| Endpoint | OpenAI-compatible, no auth | custom wire protocol | clients (opencode, LangChain, OpenAI SDK) work as-is; indistinguishable from the free legs |

**What this tier is not.** H200 here is a *quality and context* tier, not a raw-throughput
tier. Even with MTP ON (44.2 tok/s), single-stream remains ~3× below the TPU leg's
126.8 tok/s; matching that headroom is roadmap item 6 (multi-H200 TP=2), not the paid
tier in this table. The 200.4 tok/s @8-way row is **aggregate** throughput on short
128-tok bursts, not per-stream speed; "256k verified" holds for single-stream
(≈2–3 concurrent full-256k prompts, not 8).

### H200 engineering notes (measured 2026-10-30)

| Item | Measured | Note |
|---|---|---|
| KV cache footprint | 64 KB/token | hybrid attention: 16 full-attention + 48 linear-attention layers |
| KV pool | 714,258 tokens | ≈2.8 concurrent 256k prompts; 15.29 GB headroom on the 141 GB card |
| MTP draft acceptance | 53.3–54.2 % | ≈2.60–2.63 tokens per decode step (NEXTN k=3) |
| Prefill throughput | 7,831 / 9,017 / 9,395 tok/s | 7K / 28K / 113K prompts, cold card |
| Long-context | TTFT 30.8 s @254,987 tok; 42.0 s total @242,570 tok | prefill 7.8 → 9.4k tok/s over 7K → 113K |
| Cold-start breakdown | provider ~54 s · weights 26.1 s · CUDA-graph capture 65.4 s · init ~24 s | capture is the dominant single lever (snapshot/skip-capture = roadmap) |

## GLM-5.3-Flash bench — Modal B300 / H200 (2026-11-06)

Separate model from the Qwen3.8-27B trio above, on the same Modal paid tier:
**GLM-5.3-Flash**, W4A16 (humming) — 177.72 GiB across 11 safetensors shards.
Live-endpoint measurements only; no financial/accounting data included, by policy.

### B300 × 1

| Category | Metric | Result | Notes |
|---|---|---|---|
| Config | Hardware | 1× NVIDIA B300 SXM6 AC, 288 GB | tensor-parallel = 1 |
| Config | Weight quantization | GLM-5.3-Flash W4A16 (humming), 177.72 GiB / 11 shards | — |
| Config | Engine | vLLM 0.32.0 + MTP×2 | attention auto-selected → FLASHINFER_MLA; MoE = MARLIN |
| Config | Context window | 1,048,576 (1M) | reported by /v1/models |
| Startup | Cold start (3rd start) | **586 s** | allocation → /health 200; JIT cache volume warm |
| Startup | Cold start (4th start) | **~7.5–7.8 min** | via one-click desktop launcher |
| Startup | Deploy time (image cache hit) | **35 s** | 4th start |
| Startup | 1M KV on a single card | **fits** (engine allocated the 1M KV pool and booted, verified twice) | — |
| Perf | 128k prefill | **12.0 s ≈ 10.7k tok/s** (smoke); **11,378.7 tok/s** (engine log) | two independent measurements |
| Perf | Single-stream decode tok/s | not measured | pending (tier-A suite) |
| Perf | prefill ladder 7k/28k/113k | not measured | pending (tier-A suite) |
| Perf | 256k TTFT | not measured | pending (tier-A suite) |
| Perf | 840k TTFT (= 80% of 1M) | not measured | pending (tier-A, final block by design) |
| Perf | concurrency / Locust | not measured | pending (tier-B) |
| MTP | acceptance rate | **70%** (30 drafted / 21 accepted) | 15 smoke requests |
| MTP | per-position acceptance | pos0 80% (12/15), pos1 60% (9/15) | same sample |
| MTP | effective output | **~2.4 tokens/step** | — |
| MTP | engine-log short samples | 41.7% (5/12), 62.5% (5/8) | 4th start, reference only |
| Smoke | auth gate | unauthenticated → 401 (no wake); authenticated → 503 → 200 | — |
| Smoke | model list | 1 model registered, max_model_len 1,048,576 | — |
| Smoke | short request | OK (responses '323' / '4') | — |
| Smoke | KV occupancy (peak) | ≤ 1.4% during smoke | engine log |
| E2E | opencode chat | "你好" OK (two different snapshots hitting the same backend) | — |
| Incident | toolchain ICE | resolved via prebuilt JIT cache; affected 2 starts, does not affect data conclusions | on record |

### H200 × 2

| Category | Metric | Result | Notes |
|---|---|---|---|
| Config | Hardware | 2× H200 (141 GB HBM3 each) | tensor-parallel = 2 |
| Config | Weight quantization | GLM-5.3-Flash W4A16 (humming), 177.72 GiB / 11 shards (same source as B300) | — |
| Config | Engine | vLLM 0.32.0 + MTP×2 | attention explicit FLASH_ATTN_MLA_SPARSE; MoE = humming/NVRTC |
| Config | Context window | 1,048,576 (1M) | configured |
| Startup | Cold start (first start, incl. image build) | **11.3 min** | allocation → ready |
| Startup | 1M serving | verified at 128k; not stressed beyond | — |
| Perf | 128k prefill | **10.0 s ≈ 12.8k tok/s** | 2-card aggregate |
| Perf | Single-stream decode tok/s | not measured | bench deferred |
| Perf | hello latency | **0.3 s** | 28 reasoning tokens (glm45 parser active) |
| Perf | prefill ladder / 256k | not measured | bench deferred |
| Perf | concurrency / Locust | not measured | bench deferred |
| MTP | acceptance rate | **76%** (165/216) | — |
| MTP | effective output | **2.53 tokens/step** | — |
| MTP | reference (repo Qwen H200 baseline, NEXTN k=3) | 53.3–54.2% / ~2.6 tokens/step | quoted from the reference repo, not re-measured |
| E2E | opencode chat | "你好" OK (user-verified) | — |
| Deployment trait | idle behavior | platform keep-alive probing → deployed endpoint does not auto-scale-to-zero; explicit stop verified effective | — |

### Cross-check (same-metric rows)

| Metric | B300 × 1 | H200 × 2 |
|---|---|---|
| 128k prefill | 10.7–11.4k tok/s | 12.8k tok/s |
| MTP acceptance / tokens·step⁻¹ | 70% / ~2.4 | 76% / 2.53 |
| Cold start | 586 s – ~7.8 min (warm caches) | 11.3 min (first start) |
| 1M ctx single-card | fits (288 GB single card) | n/a (2 cards) |

*All "not measured" rows mark data completeness, not failures. No account, balance,
or spend information is contained in this section by design.*

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

Transferable skills:

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
- Prebuild the GPU engine offline (no GPU needed): `kaggle-tpu-lab/engine-factory/build_engine.sh`.
- Watch live events: `python app/watch_*.py`.
- Read the live endpoint/key: `python app/register_service.py status` (or `take tpu|gpu`).

Secrets never enter this repository: tokens & runtime endpoint keys are kept in
`tmp/` state files and ignored via `.gitignore`.

## Skills Used

| Skill | Where | Status |
|---|---|---|
| LLM serving (vLLM, llama.cpp, OpenAI-compat) | TPU + GPU kernels | delivered |
| Model optimization (GGUF Q4_K_M, offload, batching) | GPU engine | delivered |
| Heterogeneous accelerator scheduling (TPU/GPU via API, machine_shape) | push/probe scripts | delivered |
| Tool-call parsing & eval (qwen3_coder style) | TPU eval chain | delivered |
| Failover / health checking | controller — TPU+GPU (unhealthy→failover, healthy→recycle) | delivered |
| Service endpoint registration | `tmp/current_services.json` + `register_service.py take/status` | delivered |
| Load testing (Locust + TTFT/RPS/p95) | `loadtest/` — 5 VU run: TPU 0% error @ TTFT p95 172ms | delivered |
| Containerized engine build (Docker + CUDA devel, nvcc on CPU, reproducible) | `engine-factory/` — offline `engine.tar.gz`, pinned `engine.sha` | delivered |
| Paid Hopper serving (H200, BF16, 262k, serverless) | H200 tier — SGLang (MTP NEXTN k=3, CUDA graphs) + 2.6–3.1-min cold start + scale-to-zero ($0 idle), Modal | delivered |
| LLM observability (Langfuse/MLflow) | trace / cost-per-token / eval dashboards | planned |
| LLM-as-judge evaluation | bf16 vs Q4 comparison suite | planned |

## Roadmap

1. ~~TPU→GPU failover router~~ ✓ done (live controller, `app/controller.py`)
2. ~~Load testing~~ ✓ done (Locust, initial 5-VU baseline in `loadtest/`)
3. Saturation ramp 20-50 VU + LangGraph persona corpus
4. LLM observability (Langfuse/MLflow) + bf16-vs-Q4 eval
5. ~~Cache prebuilt llama-server binary to skip the 25-min in-kernel build~~ ✓ done — `engine-factory/` builds the identical binary in Docker (no GPU), `bootstrap_cache` feeds it in; cold build phase → cache-hit
6. H200 paid-tier throughput: ~~MTP speculative decoding~~ ✓ done (SGLang MTP NEXTN k=3, 14.8 → 44.2 tok/s); ~~≥100k-context + concurrency re-measure under SGLang~~ ✓ done 2026-10-30 (254,987-tok TTFT 30.8 s; 200.4 tok/s @8-way); next: multi-H200 TP=2, cold-start CUDA-graph-capture optimization

`kaggle-tpu-lab/` is derived from [kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab) (MIT).