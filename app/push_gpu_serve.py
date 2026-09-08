#!/usr/bin/env python3
"""Push the 2xT4 GPU serve kernel (llama.cpp, Q4_K_M) with CFG injected.
Mirrors kaggle-tpu-lab/launch.py cmd_serve but machine_shape=NvidiaTeslaT4.
Writes state to tmp/kaggle-gpu-lab.json. Never prints the Kaggle token."""
import json, re, secrets, subprocess, sys, tempfile, uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kaggle_login as kl

ROOT = Path(__file__).resolve().parent.parent
KERNEL_SRC = ROOT / "kaggle-tpu-lab" / "kernel" / "serve_qwen38_gpu.py"
STATE_FILE = ROOT / "tmp" / "kaggle-gpu-lab.json"
USER = "tentenshishi"

kl.login()

cfg = {
    "ntfy_topic": "ktl-" + uuid.uuid4().hex[:20],
    "api_key": "sk-" + secrets.token_hex(16),
    "weights_dataset": "tentenshishi/qwen3-8-27b-q4-k-m-private",
    "keepalive_min": 420,
}
src = KERNEL_SRC.read_text()
src, n = re.subn(r"^CFG = None  # __LAUNCHER_CONFIG__.*$",
                 f"CFG = {cfg!r}", src, count=1, flags=re.M)
if n != 1:
    sys.exit("serve_qwen38_gpu.py is missing the __LAUNCHER_CONFIG__ line")

slug = "qwen38-gpu-serve"
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    (td / "serve_qwen38_gpu.py").write_text(src)
    (td / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{slug}",
        "title": slug,
        "code_file": "serve_qwen38_gpu.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [cfg["weights_dataset"]],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }, indent=1))
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(td)],
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    print(out[-1600:])
    if "successfully pushed" not in out:
        sys.exit("Push failed")

STATE_FILE.write_text(json.dumps(
    {"kernel": f"{USER}/{slug}", "topic": cfg["ntfy_topic"], "api_key": cfg["api_key"]}))
print("state ->", STATE_FILE)