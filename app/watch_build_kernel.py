#!/usr/bin/env python3
"""Background-ish monitor: poll kernel status, dump output when complete."""
import re, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kaggle_login as kl

OUT = Path(__file__).resolve().parent.parent / "tmp" / "kernel_out"
KID = "tentenshishi/qwen38-private-dataset-build"

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
    time.sleep(90)

OUT.mkdir(exist_ok=True)
subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", KID, "-p", str(OUT)],
               capture_output=True, text=True)
print("--- kernel output files ---")
for f in OUT.rglob("*"):
    if f.is_file():
        print(f.relative_to(OUT), f.stat().st_size)
