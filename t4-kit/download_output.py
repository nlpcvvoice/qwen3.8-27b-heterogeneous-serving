#!/usr/bin/env python3
"""Download a completed/completed kernel output to <project>/out/<version>/.

Extracts the output zip; prints the file tree + sizes. After a cache-fill run
('--stop-on built') you expect engine.tar.gz + llamacpp.log inside."""
import argparse, json, sys, time
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kit_config import KERNEL_ID, OUT_DIR, RUNTIME
import kaggle_login as kl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--version", default="", help="output version (default: latest)")
    args = ap.parse_args()

    kl.login()
    api = kl.get_kaggle_api()
    dest = args.out
    dest.mkdir(parents=True, exist_ok=True)

    # api returns (version_no, zip_path) — download then unzip
    ver, zip_path = api.kernels_output(KERNEL_ID, path=str(dest), quiet=True)
    if not zip_path:
        sys.exit("no output available yet (kernel must be COMPLETE)")
    zip_path = Path(zip_path)
    out_dir = dest / str(ver)
    with ZipFile(zip_path) as z:
        z.extractall(out_dir)
    print(f"[t4-kit] output v{ver} -> {out_dir}")
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            print(f"   {f.relative_to(out_dir)}  ({f.stat().st_size:_} bytes)")


if __name__ == "__main__":
    main()