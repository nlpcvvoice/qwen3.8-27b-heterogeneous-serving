#!/usr/bin/env python3
"""P3: fetch the cached llama-server binary to a local out dir.

Usage:
    python fetch_cache.py [--out ./out] [--check] [--api]

Verified locally (--check) or via Kaggle API (--api). Downloaded file is
sha256-verified against manifest.json inside the dataset.
"""
import argparse
import json
import os
import subprocess
import sys
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
                            "-d", CACHE_DATASET, "-p", td, "--dir-mode", "zip"],
                           capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        print(out[-1000:])
        if r.returncode != 0:
            sys.exit("dataset download failed")

        zips = list(Path(td).glob("*.zip"))
        if not zips:
            sys.exit("no zip in download")
        args.out.mkdir(parents=True, exist_ok=True)
        bin_path = args.out / "llama-server"
        with ZipFile(zips[0]) as z:
            names = [n for n in z.namelist() if n.endswith("llama-server")]
            if not names:
                sys.exit("llama-server not found in dataset zip")
            with z.open(names[0]) as src, open(bin_path, "wb") as dst:
                os.write(dst.fileno(), src.read())
        bin_path.chmod(0o755)

    local_sha = sha256_file(bin_path)
    with ZipFile(zips[0]) as z:
        mani = [n for n in z.namelist() if n.endswith("manifest.json")]
        expect = None
        if mani:
            expect = json.loads(z.read(mani[0]))["sha256"]
        if expect is None and args.api:
            api = kl.get_kaggle_api()
            md = api.dataset_metadata(CACHE_DATASET)
            try:
                expect = md["files"][0].get("sha256")
            except Exception:
                expect = None
    if expect and local_sha != expect:
        sys.exit(f"sha MISMATCH: {local_sha} != {expect}")
    elif args.check and expect is None:
        sys.exit("no manifest to verify against; rerun with --api")

    print(f"[P3] cached llama-server  -> {bin_path}  ({bin_path.stat().st_size} bytes)")
    print(f"[P3] sha256               -> {local_sha}")
    if expect:
        print(f"[P3] verified against    {expect}")


if __name__ == "__main__":
    main()