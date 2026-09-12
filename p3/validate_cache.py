#!/usr/bin/env python3
"""P3 independent full test — verifies the engine-cache pipeline works.

Modes (any machine, zero GPU/TPU quota):
  python validate_cache.py                     # full LOCAL simulation
  python validate_cache.py --static-only       # kernel/push static assertions
  python validate_cache.py --live              # real Kaggle roundtrip (push+fetch)

Local simulation rebuilds the exact decision block the kernel runs on GPU
(cache-hit vs source-build) against a fake /kaggle/input tree and asserts:
  - cache dataset mounted  -> engine becomes llama-server, NO cmake/build cmd
  - no cache dataset       -> engine path falls through to build
  - tiny/garbage binary    -> rejected, falls through to build
Exits 0 (PASS) / 1 (FAIL). Never prints the Kaggle token.
"""
import argparse
import glob as globmod
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CACHE_DATASET,
    CACHE_SLUG,
    ELSE_BUILD_HOOK,
    KERNEL_SRC,
    SKIP_COMPILE_HOOK,
    sha256_file,
)

PUSH_SRC = Path(__file__).resolve().parent.parent / "app" / "push_gpu_serve.py"


def _build_fake_input(root: Path, slug: str, size: int, magic: bytes = b"\x7fELF") -> Path:
    inp = root / "input" / slug
    inp.mkdir(parents=True, exist_ok=True)
    blob = magic + b"\x00" * size
    (inp / "llama-server").write_bytes(blob)
    return root / "input"


def _build_fake_tar(root: Path, slug: str, size: int = 25_000_000) -> Path:
    inp = root / "input" / slug
    inp.mkdir(parents=True, exist_ok=True)
    tar = inp / "engine.tar.gz"
    import io
    import tarfile as tfm
    with tfm.open(tar, "w:gz") as tf:
        blob = b"\x7fELF" + b"\x00" * 30_000
        for name, sz in (("bin/llama-server", len(blob)),
                         ("bin/libllama-server-impl.so", 4_000_000)):
            info = tfm.TarInfo(name)
            info.size = sz
            tf.addfile(info, io.BytesIO(blob + b"\x00" * (sz - len(blob))))
    # pad to requested size so the "too small" checks behave predictably
    if tar.stat().st_size < size:
        with open(tar, "ab") as f:
            f.write(b"\x00" * (size - tar.stat().st_size))
    return tar


def decide_engine(input_dir: Path, cache_slug: str | None, force_no_cache: bool = False):
    """Mirrors the kernel's 2a cache-consume decision (tar preferred, legacy fallback)."""
    cached_tar = None
    cached_bin = None
    if cache_slug and not force_no_cache:
        for pat in (f"{cache_slug}/engine.tar.gz",
                    f"datasets/*/{cache_slug}/engine.tar.gz"):
            for hit in globmod.glob(str(input_dir / pat)):
                p = Path(hit)
                if p.is_file() and p.stat().st_size >= 20_000_000:
                    cached_tar = p
                    break
            if cached_tar:
                break
        if cached_tar:
            return "llama-server", "cache-hit", cached_tar
        for pat in (f"{cache_slug}/llama-server",
                    f"datasets/*/{cache_slug}/llama-server"):
            for hit in globmod.glob(str(input_dir / pat)):
                p = Path(hit)
                if p.is_file() and p.stat().st_size >= 1_000_000:
                    cached_bin = p
                    break
            if cached_bin:
                break
    if cached_bin:
        return "llama-server", "cache-hit", cached_bin
    return "unknown", "source-build", None


def test_local_simulation() -> None:
    # 1) cache dataset mounted (tar) -> cache-hit
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_fake_tar(root, CACHE_SLUG)
        eng, src, hit = decide_engine(root / "input", CACHE_SLUG)
        assert eng == "llama-server" and src == "cache-hit", (eng, src)
        assert hit and hit.name == "engine.tar.gz", hit.name if hit else None
        print("  [local] cache dataset mounted (tar) -> engine=llama-server (cache-hit)  OK")

    # 1b) legacy single-file still hits
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        inp = _build_fake_input(root, CACHE_SLUG, 3_000_000)
        eng, src, _ = decide_engine(inp, CACHE_SLUG)
        assert eng == "llama-server" and src == "cache-hit", (eng, src)
        print("  [local] legacy single-file cache      -> engine=llama-server (cache-hit)  OK")

    # 2) no cache dataset -> falls to source build
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        inp = _build_fake_input(root, "other-dataset", 1000)
        eng2, src2, _ = decide_engine(inp, CACHE_SLUG)
        assert src2 == "source-build", (eng2, src2)
        print("  [local] no cache dataset           -> source-build                    OK")

    # 3) tiny/garbage binary -> rejected
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        inp = _build_fake_input(root, CACHE_SLUG, 500_000)
        eng3, src3, _ = decide_engine(inp, CACHE_SLUG)
        assert src3 == "source-build", (eng3, src3)
        print("  [local] tiny/garbage binary        -> rejected -> source-build        OK")

    # 3b) tiny tar -> rejected too
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_fake_tar(root, CACHE_SLUG, size=5_000_000)
        eng3b, src3b, _ = decide_engine(root / "input", CACHE_SLUG)
        assert src3b == "source-build", (eng3b, src3b)
        print("  [local] tiny tar                   -> rejected -> source-build        OK")

    # 4) force_no_cache flag -> source build
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_fake_tar(root, CACHE_SLUG, size=25_000_000)
        eng4, src4, _ = decide_engine(root / "input", CACHE_SLUG, force_no_cache=True)
        assert src4 == "source-build", (eng4, src4)
        print("  [local] force no-cache flag        -> source-build                   OK")

    # 5) datasets/*/<slug> layout (source mount dir style)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_fake_tar(root, f"datasets/someowner/{CACHE_SLUG}", size=25_000_000)
        eng5, src5, _ = decide_engine(root / "input", CACHE_SLUG)
        assert eng5 == "llama-server" and src5 == "cache-hit", (eng5, src5)
        print("  [local] datasets/*/<slug> layout   -> cache-hit                      OK")


def test_static_assertions() -> None:
    kernel = KERNEL_SRC.read_text()
    assert SKIP_COMPILE_HOOK.search(kernel), "kernel missing # --- 2a. cache dataset hit"
    assert ELSE_BUILD_HOOK.search(kernel), "kernel missing 'elif have_toolchain:'"
    assert "engine-cache-hit" in kernel, "kernel missing engine-cache-hit publish"
    assert 'CFG.get("cache_dataset")' in kernel, "kernel must read CFG['cache_dataset']"
    assert "engine.tar.gz" in kernel, "kernel missing engine.tar.gz (tar cache path)"
    print("  [static] kernel  : skip-compile hook + else-build + cache-hit publish   OK")

    if PUSH_SRC.exists():
        push = PUSH_SRC.read_text()
        assert "CACHE_DATASET" in push, "push_gpu_serve.py missing CACHE_DATASET"
        assert "_dataset_exists" in push, "push_gpu_serve.py missing _dataset_exists()"
        assert 'cfg["cache_dataset"]' in push, "push_gpu_serve.py missing CFG injection"
        assert "dataset_sources.append(CACHE_DATASET)" in push
        print("  [static] push    : existence probe + dataset_sources injection        OK")
    else:
        print("  [static] push    : not found (optional; set PUSH_SRC= to enable)      SKIP")

    subprocess.run([sys.executable, "-m", "py_compile", str(KERNEL_SRC)],
                   check=True, capture_output=True)
    print("  [static] kernel  : py_compile clean                                     OK")


def test_live() -> None:
    import kaggle_login as kl
    from fetch_cache import main as fetch_main
    kl.login()
    api = kl.get_kaggle_api()
    try:
        md = api.dataset_status(CACHE_DATASET)
    except Exception as exc:
        print(f"  [live ] cache dataset missing -> {type(exc).__name__}; run upload first")
        sys.exit(1)
    assert md["isPrivate"] in (True, "true", "True"), "cache dataset must stay private"
    print(f"  [live ] dataset {CACHE_DATASET} isPrivate=true, {len(md['files'])} file(s)  OK")

    with tempfile.TemporaryDirectory() as td:
        sys.argv = ["fetch_cache.py", "--out", str(Path(td) / "out"), "--check"]
        fetch_main()
    print("  [live ] fetch/download + sha verify complete                        OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()

    print("[P3 validate] engine-cache full test")
    if not args.static_only:
        test_local_simulation()
    test_static_assertions()
    if args.live:
        test_live()
    print("[P3 validate] ALL PASS")


if __name__ == "__main__":
    main()