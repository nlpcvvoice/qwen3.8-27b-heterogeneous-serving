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
| Single-stream decode | 126.8 tok/s | 13.6 tok/s |
| 4 concurrent requests | (vLLM batching) | 4 req / 4.4 s, 0 errors |
| Tool-call emission | `list_files({"path":"."})` ✓ | — |
| Startup to ready | ~20 min (weight load + XLA compile) | ~95 s (weights offload) |

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
| Load testing (Locust + persona corpus) | in progress |
| LLM observability (Langfuse/MLflow) | planned |
| LLM-as-judge evaluation | planned |

## Roadmap

1. TPU→GPU failover router (local controller)
2. Load testing with Locust + synthetic persona corpus (QPS/TTFT/p50/p95)
3. LLM observability (Langfuse/MLflow) + bf16-vs-Q4 eval
4. Cache prebuilt llama-server binary to skip the 25-min in-kernel build

`kaggle-tpu-lab/` is derived from [kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab) (MIT).