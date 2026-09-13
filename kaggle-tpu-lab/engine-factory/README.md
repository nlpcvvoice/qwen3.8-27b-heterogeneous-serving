# engine-factory — offline containerized build of the GPU inference engine

Appendix to the heterogeneous-serving stack. Produces the CUDA `llama-server`
engine used by the T4 GPU kernel (`kaggle-tpu-lab/kernel/serve_qwen38_gpu_mtp.py`)
outside of Kaggle, eliminating the ~25 min in-kernel cmake build on cold kernels.

## Rationale

- **No GPU required** — nvcc emits `sm_75` target code; compilation is CPU-bound.
- **Reproducible** — same base image + identical cmake flags + a pinned
  `llama.cpp` commit (`engine.sha`) yield a byte-stable artifact.
- **Zero Kaggle cost** — the build consumes only local (Docker) cycles, so no
  GPU-session time or quota is burnt compiling with the accelerator idle.
- **Drop-in** — the artifact is consumed through the existing
  `bootstrap_cache.py` / `unpack_engine()` pipeline; no kernel changes required.

## Build invocation

```bash
# Default: llama.cpp master (same source lineage as the in-kernel build)
./build_engine.sh

# Custom output directory / pinned revision
./build_engine.sh /path/to/out --commit <sha-or-master>
```

### Artifacts (written to `OUT`)

| File | Content |
|---|---|
| `engine.tar.gz` | kernel-required layout (`bin/`, plus `lib/` when present); validated by `package.sh` |
| `engine.sha` | resolved `llama.cpp` commit for version pinning |

> First build pulls `nvidia/cuda:12.4.0-devel-ubuntu22.04` (~5 GB) and compiles in
> ~20-40 min on 4 vCPU. Later builds reuse Docker layer caches and package in seconds.

## Reproducing the in-kernel build

| Parameter | In-kernel (`serve_qwen38_gpu_mtp.py` §2) | engine-factory |
|---|---|---|
| Source | git clone `ggml-org/llama.cpp` (master) | same, pinnable via `--commit` |
| Flags | `GGML_CUDA=ON` · `FORCE_DMMV` · `CCACHE=OFF` · Release · arch `sm_75` · `LLAMA_NATIVE=OFF` | identical |
| Output | `WORK/engine.tar.gz` (`bin/`+`lib/`) | `OUT/engine.tar.gz` + `engine.sha` |
| Runtime cost | GPU-session minutes, card idle | none |

### Required build-time note

`-DBUILD_SHARED_LIBS=OFF` is mandatory. With shared ggml the CUDA-driver
dependency is embedded in `libggml-cuda.so`, so the `llama-server` link step
does not resolve the `cu*` driver entry points and fails with undefined-symbol
errors. Statically linking ggml resolves the driver at the final link.

## Integration with the kernels

```
OUT/engine.tar.gz
   → upload to the private cache dataset (e.g. llama-server-qwen38-cache)
   → push script auto-attaches the dataset
   → kernel: find_input → unpack_engine() → engine-cache-hit, ready in seconds
```

The unpacked engine must be uploaded via the same `cache-upload` flow already
used for in-kernel builds; nothing else in the runtime path changes.

## Validation

| Check | Method |
|---|---|
| Archive layout | `tar tzf engine.tar.gz` contains `bin/llama-server` |
| Target architecture | build log shows `arch=compute_75`; binary links CUDA runtimes |
| Dynamic dependencies | `package.sh` runs `readelf`: NEEDED `libcublas.so.12`, `libcuda.so.1` (present on the Kaggle runtime image) |
| Reproducibility | `engine.sha` pins the commit; repeat build with same args → identical artifact |
| Live boot | one kernel run; confirm `engine-cache-hit` (needs a GPU-enabled session) |

## Operational notes

- The container links against the CUDA **driver stub** at build time; the real
  driver is supplied by the Kaggle host at runtime.
- The base image (Ubuntu 22.04 + CUDA 12.x) is ABI-matched to the Kaggle kernel
  runtime image.
- Artifacts stay under `OUT/` (gitignored); only the build scripts, Dockerfile
  and this document are committed.