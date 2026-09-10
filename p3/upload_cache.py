#!/usr/bin/env python3
"""P3: upload a built llama-server binary to the private engine-cache dataset.

Usage:
    python upload_cache.py /path/to/llama-server [--sha <sha256>] [--msg "build note"]

Writes to the private dataset CACHE_DATASET (creates on first run, versions
thereafter). Never prints the Kaggle token. Datasets push is free (no GPU/TPU
quota) but counts against Kaggle's data-upload quota.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_login as kl  # noqa: E402
from common import CACHE_DATASET, build_cache_stage, sha256_file  # noqa: E402


def exists(api) -> bool:
    try:
        api.dataset_status(CACHE_DATASET)
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", type=Path, help="llama-server binary to cache")
    ap.add_argument("--sha", default="", help="expected sha256 (default: computed)")
    ap.add_argument("--msg", default="update-engine", help="dataset version message")
    args = ap.parse_args()

    if not args.binary.is_file():
        sys.exit(f"binary not found: {args.binary}")
    if args.binary.stat().st_size < 1_000_000:
        sys.exit("binary too small (<1MB); looks wrong.")

    kl.login()
    api = kl.get_kaggle_api()

    stage = build_cache_stage(args.binary, want_sha=args.sha or None)
    sha = sha256_file(stage / "llama-server")
    meta = {
        "id": CACHE_DATASET,
        "title": "llama-server (Qwen3.8-27B) engine cache",
        "subtitle": "prebuilt CUDA llama-server binary to skip 25-min cmake build",
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

    print(f"[P3] uploaded llama-server sha256={sha}")
    print(f"[P3] {CACHE_DATASET} versioned (private)")


if __name__ == "__main__":
    main()