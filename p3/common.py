#!/usr/bin/env python3
"""Shared P3 engine-cache helpers. No secrets here; auth lives in kaggle_login.py."""
import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path

P3_ROOT = Path(__file__).resolve().parent
CACHE_DATASET = "tentenshishi/llama-server-qwen38-cache"
CACHE_SLUG = CACHE_DATASET.split("/")[-1]
SKIP_COMPILE_HOOK = re.compile(r"# --- 2a\. cache dataset hit")
ELSE_BUILD_HOOK = re.compile(r"elif have_toolchain:")
MIN_BYTES = 1_000_000  # sanity floor for a legacy llama-server file
MIN_TAR_BYTES = 20_000_000  # sanity floor for engine.tar.gz (bin+lib bundle)
KERNEL_SRC = P3_ROOT / "kernel" / "serve_qwen38_gpu.py"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def is_elf(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == b"\x7fELF"


def reusable_tmp() -> Path:
    d = P3_ROOT / "tmp"
    d.mkdir(exist_ok=True)
    return d


def sqlite_free_upload_dir() -> Path:
    return reusable_tmp()


def build_cache_stage(artifact: Path, want_sha: str | None = None) -> Path:
    """Pack engine.tar.gz + manifest.json into a temp dir for kaggle dataset push.

    The artifact is the full runtime bundle (BUILD/bin + BUILD/lib tarball)
    produced by the kernel; it MUST contain the thin llama-server launcher plus
    its impl/shared libs (e.g. libllama-server-impl.so, libggml*.so).
    """
    stage = Path(tempfile.mkdtemp(prefix="p3-stage-", dir=sqlite_free_upload_dir()))
    dst = stage / "engine.tar.gz"
    shutil.copy2(artifact, dst)
    sha = want_sha or sha256_file(dst)
    (stage / "manifest.json").write_text(
        f'{{"sha256":"{sha}","size":{dst.stat().st_size},"toolchain":"CUDA-75",'
        f'"artifact":"engine.tar.gz","gzip":true}}\n')
    return stage