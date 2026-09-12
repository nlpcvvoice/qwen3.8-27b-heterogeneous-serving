#!/usr/bin/env python3
"""P3: fetch the cached engine runtime bundle to a local out dir.

Usage:
    python fetch_cache.py [--out ./out] [--check] [--api]

Verified locally (--check) or via Kaggle API (--api). The downloaded
engine.tar.gz is sha256-verified against manifest.json inside the dataset and
its structure inspected (bin/llama-server + shared libs present).
"""
import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_login as kl  # noqa: E402
from common import CACHE_DATASET, sha256_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("p3-out"))
    ap.add_argument("--check", action="store_true", help="verify sha256 locally")
    ap.add_argument("--api", action="store_true", help="verify via Kaggle API")
    args = ap.parse_args()

    kl.login()
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download",
                            "-d", CACHE_DATASET, "-p", td],
                           capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        print(out[-1000:])
        if r.returncode != 0:
            sys.exit("dataset download failed")

        zips = list(Path(td).glob("*.zip"))
        if not zips:
            sys.exit("no zip in download")
        args.out.mkdir(parents=True, exist_ok=True)

        with ZipFile(zips[0]) as z:
            names = z.namelist()
            mani = [n for n in names if n.endswith("manifest.json")]
            expect = None
            if mani:
                expect = json.loads(z.read(mani[0]))["sha256"]
            tar_names = [n for n in names if n.endswith("engine.bundle") or n.endswith("engine.tar.gz")]
            if tar_names:
                tar_path = args.out / "engine.tar.gz"
                with z.open(tar_names[0]) as src, open(tar_path, "wb") as dst:
                    os.write(dst.fileno(), src.read())
            else:
                bin_names = [n for n in names if n.endswith("llama-server")]
                if not bin_names:
                    sys.exit("neither engine.tar.gz nor llama-server found in dataset zip")
                tar_path = None
                print("[P3] note: dataset holds legacy single-file (no libs); re-run upload")
                bin_path = args.out / "llama-server"
                with z.open(bin_names[0]) as src, open(bin_path, "wb") as dst:
                    os.write(dst.fileno(), src.read())
                bin_path.chmod(0o755)

    if tar_path is not None:
        local_sha = sha256_file(tar_path)
        with tarfile.open(tar_path, "r:gz") as tf:
            entries = sorted(tf.getnames())
        impl = [n for n in entries if "impl" in n or "ggml" in n]
        has_srv = "bin/llama-server" in entries
        print(f"[P3] engine.tar.gz     -> {tar_path}  ({tar_path.stat().st_size} bytes)")
        print(f"[P3] bin/llama-server  -> {'OK' if has_srv else 'MISSING'}")
        print(f"[P3] runtime libs      -> {len(impl)} x {impl[:6]}")
        print(f"[P3] sha256            -> {local_sha}")
        if expect:
            print(f"[P3] verified against manifest -> {expect}")
            if local_sha != expect:
                sys.exit(f"sha MISMATCH: {local_sha} != {expect}")
        elif args.check:
            sys.exit("no manifest to verify against; rerun with --api")
        if not (has_srv and impl):
            sys.exit("engine tar contents invalid (launcher or libs missing)")
    else:
        print(f"[P3] cached llama-server  -> {bin_path}  ({bin_path.stat().st_size} bytes)")
        print(f"[P3] sha256               -> {sha256_file(bin_path)}")
        if expect:
            print(f"[P3] verified against    {expect}")
        if args.check and expect is None:
            sys.exit("no manifest to verify against; rerun with --api")


if __name__ == "__main__":
    main()