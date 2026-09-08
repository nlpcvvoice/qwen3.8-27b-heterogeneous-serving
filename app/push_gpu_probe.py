#!/usr/bin/env python3
"""Push the GPU probe kernel (machine_shape=NvidiaTeslaT4) and print status."""
import json, re, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kaggle_login as kl

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "tmp" / "kernel_pkg_gpu_probe"
PATH_P = PKG / "probe_gpu_kernel.py"
PATH_M = PKG / "kernel-metadata.json"

USER = "tentenshishi"
SLUG = "qwen38-gpu-probe"

kl.login()

meta = {"id": f"{USER}/{SLUG}", "title": SLUG,
        "code_file": "probe_gpu_kernel.py",
        "language": "python", "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        "model_sources": []}
PATH_M.write_text(json.dumps(meta, indent=1))

r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(PKG)],
                   capture_output=True, text=True)
out = (r.stdout or "") + (r.stderr or "")
print(out[-2000:])
if "successfully pushed" not in out:
    sys.exit("push failed")

kid = f"{USER}/{SLUG}"
for _ in range(3):
    rr = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", kid],
                        capture_output=True, text=True)
    o = (rr.stdout or "") + (rr.stderr or "")
    m = re.search(r'"KernelWorkerStatus\.(\w+)"', o)
    print(time.strftime("%H:%M:%S"), m.group(1) if m else o.strip()[:120])
    time.sleep(5)
print("pushed:", kid)