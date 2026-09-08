# Qwen3.8-27B — Heterogeneous Serving (TPU v5e + GPU T4)

Free high-end open-source LLM serving for agent/tool workloads: the same
**Qwen3.8-27B** model exposed as an OpenAI-compatible API from **two** free
accelerator backends, with automatic failover TPU → GPU.

```
                    clients (opencode, curl, agents...)
                     ┌──────────────┐   OpenAI-compatible /v1
                     │      router  │   (planned: local controller)
                     └──────┬───┬───┘
                    primary │   │ fallback
                            ▼   ▼
   TPU v5e-8 ─── vLLM (bf16, 262k ctx)      GPU 2xT4 ─── llama.cpp (Q4_K_M, llama-server)
   ~126.8 tok/s · queues 0-3h               ~13.6 tok/s · instant slots
```

## Why / use case

- Self-host a flagship open-weight model at **\$0 inference cost** for
  evaluation, agent tool-calling and long-context workloads — versus per-token
  API pricing.
- The GPU leg acts as an on-demand **fallback** when the TPU queue/session is
  unavailable, so the service has a failover story, not a single point.

## Model architecture & trade-offs

| | TPU v5e-8 (primary) | GPU 2xT4 (fallback) |
|---|---|---|
| Precision | bf16 full | GGUF Q4_K_M (4-bit) |
| Engine | vLLM (XLA/TPU) | llama.cpp (CUDA sm_75) |
| Context | 262144 | 98304 total / 24576 per slot (KV q8_0) |
| Throughput | 126.8 tok/s decode | 13.6 tok/s decode |
| Concurrency | vLLM continuous batching | 4 parallel slots |
| Queue latency | 0-3 h (shared free pool) | instant |
| Why | max quality*throughput | any-time availability |

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
| Availability | TPU→GPU failover router (planned, repository roadmap) |
| Cost & quotas | separate TPU (~20h/wk) and GPU (~30h/wk) free quotas; keepalive-limited sessions |

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

Secrets never enter this repository: tokens & runtime endpoint keys are kept in
`tmp/` state files and ignored via `.gitignore`.

## Skills Used

| Skill | Where |
|---|---|
| LLM serving (vLLM, llama.cpp, OpenAI-compat) | TPU + GPU kernels |
| Model optimization (GGUF Q4_K_M, offload, batching) | GPU engine |
| Heterogeneous accelerator scheduling (TPU/GPU via API, machine_shape) | push/probe scripts |
| Tool-call parsing & eval (qwen3_coder style) | TPU eval chain |
| Failover / health checking | router (roadmap) |
| Load testing (Locust + TTFT/RPS/p95) | `loadtest/` — 5 VU run: TPU 0% error @ TTFT p95 172ms | ✓ |
| LLM observability (Langfuse/MLflow) | planned |
| LLM-as-judge evaluation | planned |

## Roadmap

1. TPU→GPU failover router (local controller)
2. ~~Load testing~~ ✓ done (Locust, initial 5-VU baseline in `loadtest/`)
3. Saturation ramp 20-50 VU + LangGraph persona corpus
4. LLM observability (Langfuse/MLflow) + bf16-vs-Q4 eval
5. Cache prebuilt llama-server binary to skip the 25-min in-kernel build

`kaggle-tpu-lab/` is derived from [kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab) (MIT).