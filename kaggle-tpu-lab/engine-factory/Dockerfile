# Engine factory: prebuild llama-server (CUDA, arch 75) offline -> engine.tar.gz
# Reproduces the in-kernel source build of kaggle-tpu-lab/kernel/serve_qwen38_gpu_mtp.py
# so Kaggle kernels boot with engine-cache-hit instead of a ~25 min cmake build.
# Building needs NO GPU: nvcc compiles with CPU only.
ARG CUDA_IMG=12.4.0-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_IMG}

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git ca-certificates cmake gcc g++ make tar \
 && rm -rf /var/lib/apt/lists/*

# Default = master (same as the in-kernel build); pin a commit hash to freeze a version.
ARG LLAMA_COMMIT=master
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /src \
 && cd /src \
 && case "$LLAMA_COMMIT" in \
      master) ;; \
      *) git fetch --depth 1 origin "$LLAMA_COMMIT" \
         && git checkout --quiet FETCH_HEAD ;; \
    esac

# Flags locked to the kernel's cmake line (serve_qwen38_gpu_mtp.py, section 2):
#   GGML_CUDA=ON / FORCE_DMMV / CCACHE=OFF / Release / ARCHITECTURES=75 / NATIVE=OFF
# BUILD_SHARED_LIBS=OFF is REQUIRED: with shared ggml the CUDA driver dep lives in
# libggml-cuda.so and the llama-server link gets no libcuda -> undefined cu* refs.
RUN cd /src \
 && cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DGGML_CUDA_FORCE_DMMV=ON -DGGML_CCACHE=OFF \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_NATIVE=OFF \
 && cmake --build build --config Release -j "$(nproc)" --target llama-server

COPY package.sh /package.sh
RUN chmod +x /package.sh
ENTRYPOINT ["/package.sh"]