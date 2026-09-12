#!/usr/bin/env python3
"""P3: upload the engine runtime bundle to the private engine-cache dataset.

Usage:
    python upload_cache.py /path/to/engine.tar.gz [--sha <sha256>] [--msg "note"]

The artifact is the tarball (BUILD/bin + BUILD/lib) produced by the GPU kernel
(source-build path). We validate it is a gzip tar containing bin/llama-server
and at least one shared runtime lib (libllama-server-impl.so / libggml*.so) —
the thin 17 KB launcher alone is NOT enough. Pushes a private Kaggle dataset
(free, no GPU/TPU quota). Never prints the Kaggle token.
"""
import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_login as kl  # noqa: E402
from common import CACHE_DATASET, MIN_TAR_BYTES, build_cache_stage, sha256_file  # noqa: E402


def exists(api) -> bool:
    try:
        api.dataset_status(CACHE_DATASET)
        return True
    except Exception:
        return False


def validate_engine_tar(path: Path) -> list:
    """Return [ok, problems] after checking tar structure contains launcher + libs."""
    probs = []
    try:
        with tarfile.open(path, "r:gz") as tf:
            names = tf.getnames()
    except Exception as exc:  # noqa: BLE001
        return False, [f"not a gzip tar: {exc.__class__.__name__}"]
    has_srv = any(n == "bin/llama-server" for n in names)
    has_libs = any((n.startswith("bin/") or n.startswith("lib/")) and
                   any(k in n for k in ("libllama-server-impl.so", "libggml")) for n in names)
    if not has_srv:
        probs.append("bin/llama-server missing")
    if not has_libs:
        probs.append("no libllama-server-impl.so / libggml*.so inside (thin shell?)")
    return (not probs), probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", type=Path, help="engine.tar.gz (bin+lib bundle) to cache")
    ap.add_argument("--sha", default="", help="expected sha256 (default: computed)")
    ap.add_argument("--msg", default="update-engine", help="dataset version message")
    args = ap.parse_args()

    if not args.engine.is_file():
        sys.exit(f"engine tarball not found: {args.engine}")
    if args.engine.stat().st_size < MIN_TAR_BYTES:
        sys.exit(f"engine tarball too small (<{MIN_TAR_BYTES // 1_000_000} MB); looks wrong.")
    ok, probs = validate_engine_tar(args.engine)
    if not ok:
        sys.exit("engine tarball invalid: " + "; ".join(probs))

    kl.login()
    api = kl.get_kaggle_api()

    stage = build_cache_stage(args.engine, want_sha=args.sha or None)
    sha = sha256_file(stage / "engine.bundle")
    meta = {
        "id": CACHE_DATASET,
        "title": "llama-server (Qwen3.8-27B) engine cache",
        "subtitle": "prebuilt CUDA llama-server runtime bundle (bin+lib) to skip 25-min cmake build",
        "isPrivate": "true",
        "licenses": [{"name": "other"}],
    }
    (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))

    if exists(api):
        cmd = ["kaggle", "datasets", "version", "-p", str(stage), "-m", args.msg, "--dir-mode", "zip"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(stage), "--dir-mode", "zip"]
    r = subprocess.run([sys.executable, "-m", *cmd], capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    print(out[-1500:])
    if r.returncode != 0:
        sys.exit("dataset push failed")

    print(f"[P3] uploaded engine.tar.gz sha256={sha}")
    print(f"[P3] {CACHE_DATASET} versioned (private)")


if __name__ == "__main__":
    main()