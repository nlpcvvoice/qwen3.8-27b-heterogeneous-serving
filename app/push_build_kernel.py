#!/usr/bin/env python3
"""Push + monitor the one-time private-dataset build kernel (CPU, no TPU)."""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kaggle_login as kl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PKG = ROOT / "tmp" / "kernel_pkg"
OUT = ROOT / "tmp" / "kernel_out"
USER = "tentenshishi"
SLUG = "qwen38-private-dataset-build"


def main():
    kl.login()
    src = (HERE / "build_private_dataset.py").read_text()
    PKG.mkdir(parents=True, exist_ok=True)
    (PKG / "build_private_dataset.py").write_text(src)
    meta = {"id": f"{USER}/{SLUG}", "title": SLUG,
            "code_file": "build_private_dataset.py",
            "language": "python", "kernel_type": "script", "is_private": "true",
            "enable_gpu": "false", "enable_tpu": "false", "enable_internet": "true",
            "dataset_sources": ["rahim3/qwen3-8-27b-bf16"],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (PKG / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))

    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(PKG)],
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    print(out[-2000:])
    if "successfully pushed" not in out:
        sys.exit("push failed")

    kid = f"{USER}/{SLUG}"
    while True:
        rr = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", kid],
                            capture_output=True, text=True)
        o = (rr.stdout or "") + (rr.stderr or "")
        m = re.search(r'"KernelWorkerStatus\.(\w+)"', o)
        st = m.group(1) if m else (o.strip().splitlines()[0] if o.strip() else "UNKNOWN")
        print(time.strftime("%H:%M:%S"), st, flush=True)
        if st in ("COMPLETE", "ERROR", "CANCELED", "CANCELACKNOWLEDGED"):
            break
        time.sleep(60)

    OUT.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", kid, "-p", str(OUT)],
                   capture_output=True, text=True)
    print("--- kernel output files ---")
    for f in OUT.rglob("*"):
        if f.is_file():
            print(f.relative_to(OUT), f.stat().st_size)


if __name__ == "__main__":
    main()