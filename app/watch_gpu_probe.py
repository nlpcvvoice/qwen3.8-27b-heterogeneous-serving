#!/usr/bin/env python3
"""Background monitor: poll GPU probe kernel status, dump output when terminal."""
import re, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kaggle_login as kl

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tmp" / "kernel_out_gpu_probe"
KID = "tentenshishi/qwen38-gpu-probe"

kl.login()
while True:
    rr = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", KID],
                        capture_output=True, text=True)
    o = (rr.stdout or "") + (rr.stderr or "")
    m = re.search(r'"KernelWorkerStatus\.(\w+)"', o)
    st = m.group(1) if m else "UNKNOWN"
    print(time.strftime("%H:%M:%S"), st, flush=True)
    if st in ("COMPLETE", "ERROR", "CANCELED", "CANCELACKNOWLEDGED"):
        break
    time.sleep(60)

OUT.mkdir(exist_ok=True)
subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", KID, "-p", str(OUT)],
               capture_output=True, text=True)
print("--- kernel output files ---")
for f in OUT.rglob("*"):
    if f.is_file():
        print(f.relative_to(OUT), f.stat().st_size)