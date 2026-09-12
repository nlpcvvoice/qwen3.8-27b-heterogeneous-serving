#!/usr/bin/env python3
"""t4-kit offline self-test (zero GPU/TPU quota, no network to Kaggle).

  t4-kit/self_test.py                -> full local test
  t4-kit/self_test.py --static-only  -> py_compile + static assertions only

Simulates the kernel's cache-consume decision against a fake /kaggle/input tree:
  - engine.tar.gz present (>20 MB)  -> cache-hit (no cmake build)
  - legacy llama-server present      -> cache-hit (fallback)
  - tiny/garbage tar or binary       -> rejected -> source-build
Exits 0 (PASS) / 1 (FAIL). Never touches token or network."""
import argparse, glob as globmod, os, re, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kit_config import KERNEL_SRC

KIT_PY = sorted(p for p in ROOT.rglob("*.py") if "__pycache__" not in str(p))

CACHE_SLUG = "llama-server-qwen38-cache"
SKIP_HOOK = re.compile(r"# --- 2a\. cache dataset hit")
ELSE_HOOK = re.compile(r"elif have_toolchain:")


def _fake_bin(root, slug, size, magic=b"\x7fELF"):
    inp = root / "input" / slug
    inp.mkdir(parents=True, exist_ok=True)
    (inp / "llama-server").write_bytes(magic + b"\x00" * size)
    return root / "input"


def _fake_tar(root, slug, size=25_000_000):
    import io
    import tarfile
    inp = root / "input" / slug
    inp.mkdir(parents=True, exist_ok=True)
    tar = inp / "engine.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        blob = b"\x7fELF" + b"\x00" * 30_000
        for name, sz in (("bin/llama-server", len(blob)),
                         ("bin/libllama-server-impl.so", 4_000_000)):
            info = tarfile.TarInfo(name)
            info.size = sz
            tf.addfile(info, io.BytesIO(blob + b"\x00" * (sz - len(blob))))
    if tar.stat().st_size < size:
        with open(tar, "ab") as f:
            f.write(b"\x00" * (size - tar.stat().st_size))
    return tar


def decide_engine(input_dir, cache_slug, force_no_cache=False):
    cached_tar = cached_bin = None
    if cache_slug and not force_no_cache:
        for pat in (f"{cache_slug}/engine.tar.gz", f"datasets/*/{cache_slug}/engine.tar.gz"):
            for hit in globmod.glob(str(input_dir / pat)):
                p = Path(hit)
                if p.is_file() and p.stat().st_size >= 20_000_000:
                    cached_tar = p
                    break
            if cached_tar:
                break
        if cached_tar:
            return ("llama-server", "cache-hit", cached_tar.name)
        for pat in (f"{cache_slug}/llama-server", f"datasets/*/{cache_slug}/llama-server"):
            for hit in globmod.glob(str(input_dir / pat)):
                p = Path(hit)
                if p.is_file() and p.stat().st_size >= 1_000_000:
                    cached_bin = p
                    break
            if cached_bin:
                break
    if cached_bin:
        return ("llama-server", "cache-hit", cached_bin.name)
    return ("unknown", "source-build", None)


def test_local_simulation():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); _fake_tar(root, CACHE_SLUG)
        eng, src, name = decide_engine(root / "input", CACHE_SLUG)
        assert (eng, src) == ("llama-server", "cache-hit"), (eng, src, name)
        print("  [local] engine.tar.gz mounted      -> cache-hit (skip build)          OK")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td); _fake_bin(root, CACHE_SLUG, 3_000_000)
        eng, src, _ = decide_engine(root / "input", CACHE_SLUG)
        assert (eng, src) == ("llama-server", "cache-hit"), (eng, src)
        print("  [local] legacy single-file cache   -> cache-hit                      OK")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td); _fake_bin(root, "other-dataset", 1000)
        eng, src, _ = decide_engine(root / "input", CACHE_SLUG)
        assert src == "source-build", (eng, src)
        print("  [local] no cache dataset           -> source-build                   OK")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td); _fake_bin(root, CACHE_SLUG, 500_000)
        es = decide_engine(root / "input", CACHE_SLUG)
        assert es[1] == "source-build", es
        print("  [local] tiny/garbage binary        -> rejected -> source-build       OK")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td); _fake_tar(root, CACHE_SLUG, size=5_000_000)
        es = decide_engine(root / "input", CACHE_SLUG)
        assert es[1] == "source-build", es
        print("  [local] tiny tar                   -> rejected -> source-build       OK")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td); _fake_tar(root, CACHE_SLUG)
        es = decide_engine(root / "input", CACHE_SLUG, force_no_cache=True)
        assert es[1] == "source-build", es
        print("  [local] force no-cache flag        -> source-build                   OK")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td); _fake_tar(root, f"datasets/someowner/{CACHE_SLUG}")
        es = decide_engine(root / "input", CACHE_SLUG)
        assert es[1] == "cache-hit", es
        print("  [local] datasets/*/<slug> layout   -> cache-hit                      OK")


def test_static():
    k = KERNEL_SRC.read_text()
    for needle, msg in (
        (SKIP_HOOK, "kernel missing '--- 2a. cache dataset hit'"),
        (ELSE_HOOK, "kernel missing 'elif have_toolchain:'"),
        ("engine-cache-hit", "kernel missing engine-cache-hit publish"),
        ("engine.tar.gz", "kernel missing engine.tar.gz (tar cache path)"),
        ("__LAUNCHER_CONFIG__", "kernel missing CFG injection hook"),
        ("NvidiaTeslaT4", "kernel should target 2xT4 (NvidiaTeslaT4)"),
    ):
        assert (needle.search(k) if hasattr(needle, "search") else needle in k), msg
    print("  [static] kernel  : hooks + tar-cache + T4 target                        OK")

    push = (ROOT / "push_job.py").read_text()
    for needle, msg in (
        ("CACHE_DATASET", "push missing CACHE_DATASET"), ("machine_shape", "push missing machine_shape"),
        ("__LAUNCHER_CONFIG__", "push missing CFG injection"), ("dataset_sources", "push missing dataset_sources"),
    ):
        assert needle in push, msg
    print("  [static] push    : cache probe + CFG injection + T4 metadata            OK")

    for p in (Path(__file__).resolve().parent.glob("*.py")):
        subprocess.run([sys.executable, "-m", "py_compile", str(p)], check=True,
                       capture_output=True)
    print(f"  [static] compile : {len(list(Path(__file__).resolve().parent.glob('*.py')))} kit modules py_compile clean   OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-only", action="store_true")
    args = ap.parse_args()
    print("[t4-kit self-test] T4 job pipeline (offline)")
    if not args.static_only:
        test_local_simulation()
    test_static()
    print("[t4-kit self-test] ALL PASS")


if __name__ == "__main__":
    main()