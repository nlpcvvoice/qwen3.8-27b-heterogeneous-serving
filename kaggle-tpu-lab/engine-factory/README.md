# engine-factory — offline containerized llama-server build (no GPU)

Pre-builds the GPU engine (`llama-server`, CUDA sm_75) outside Kaggle, so kernels
boot with `engine-cache-hit` instead of a ~25 min in-kernel cmake build.

- Compilation is CPU-only (nvcc emits sm_75 target code; no GPU required).
- Same base image + same flags → reproducible artifact, pinned by `engine.sha`.
- Artifact ships through the existing `bootstrap_cache.py` / `unpack_engine()`
  pipeline unchanged.

## Usage

```bash
# Default: master branch (same source as the in-kernel build), output to ./out
./build_engine.sh

# Custom output dir / pinned version
./build_engine.sh /path/to/out --commit <sha-or-master>

# Artifacts (in OUT)
#   engine.tar.gz   kernel-required layout (bin/ + lib/), validated by package.sh
#   engine.sha      resolved llama.cpp commit for versioning
```

> First run pulls `nvidia/cuda:12.4.0-devel-ubuntu22.04` (~5 GB) and compiles
> (~20-40 min on 4 vCPU). Subsequent runs hit Docker build cache and package in seconds.

## Correspondence to the kernel source build

| Item | In-kernel (serve_qwen38_gpu_mtp.py §2) | Factory |
|---|---|---|
| Source | git clone ggml-org/llama.cpp master | same repo (master, pinnable via `--commit`) |
| Flags | GGML_CUDA=ON / FORCE_DMMV / CCACHE=OFF / Release / arch sm_75 / NATIVE=OFF | identical |
| Output | `WORK/engine.tar.gz` (bin+lib) | `/out/engine.tar.gz` + `engine.sha` |
| Cost | Kaggle GPU-session time, card idle | local CPU + Docker, zero Kaggle quota |

`-DBUILD_SHARED_LIBS=OFF` is required: with shared ggml the CUDA-driver dependency
lives inside `libggml-cuda.so` and the `llama-server` link gets no `libcuda`,
causing undefined `cu*` references. Static ggml resolves them at the final link.

## Feeding the kernel (reuses the existing pipeline)

```
factory → OUT/engine.tar.gz → upload to private dataset (e.g. llama-server-qwen38-cache)
       → push script auto-attaches it → kernel find_input → unpack_engine() → ready
```

## Validation checklist

| Check | Method |
|---|---|
| tar layout | `tar tzf engine.tar.gz` contains `bin/llama-server` (and `lib/` when present) |
| target arch sm_75 | build logs confirm `arch=compute_75`; binary links CUDA runtime libs |
| dynamic libs | package.sh readelf check: NEEDED libcublas.so.12 / libcuda.so.1 (present on Kaggle runtime) |
| reproducibility | `engine.sha` records the commit; rebuild with same args → same artifact |
| live boot | one kernel run, watch `engine-cache-hit` (runtime check, requires GPU grant) |

## Notes

- The CUDA driver is provided by the Kaggle host at runtime; the container only
  links against the driver stub at build time.
- The base image (Ubuntu 22.04 + CUDA 12.x) matches the Kaggle kernel runtime
  image for ABI compatibility.
- Artifacts stay in `OUT/` (gitignored); only the build scripts and this doc are committed.