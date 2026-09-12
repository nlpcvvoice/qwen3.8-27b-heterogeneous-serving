#!/usr/bin/env python3
"""Push a 2xT4 GPU kernel (llama.cpp, Q4_K_M, Qwen3.8-27B) with CFG injected.
Creates the engine-cache dataset probe to optionally skip the ~25-min cmake build.
Writes state to <project>/tmp/t4-kit/t4-job.json. Never prints the Kaggle token."""
import argparse, json, re, secrets, subprocess, sys, tempfile, uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kit_config import (
    CACHE_DATASET, KERNEL_ID, KERNEL_SLUG, KERNEL_SRC, PARENT_DIR,
    STATE_FILE, T4_OWNER, WEIGHTS_DATASET, RUNTIME,
)
import kaggle_login as kl  # noqa: E402


def _dataset_exists(ref: str) -> bool:
    r = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "status", ref],
                       capture_output=True, text=True)
    return r.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keepalive-min", type=int, default=120,
                    help="serve-time keepalive in minutes (quota-safe default: 120)")
    ap.add_argument("--no-cache", action="store_true",
                    help="never attach the engine-cache dataset (force source build)")
    ap.add_argument("--dry", action="store_true", help="print CFG and exit without pushing")
    args = ap.parse_args()

    kl.login()
    cfg = {
        "ntfy_topic": "t4k-" + uuid.uuid4().hex[:18],
        "api_key": "sk-" + secrets.token_hex(16),
        "weights_dataset": WEIGHTS_DATASET,
        "keepalive_min": args.keepalive_min,
    }
    dataset_sources = [WEIGHTS_DATASET]
    if not args.no_cache and _dataset_exists(CACHE_DATASET):
        cfg["cache_dataset"] = CACHE_DATASET
        dataset_sources.append(CACHE_DATASET)
        print(f"[t4-kit] cache dataset found -> skip 25-min build: {CACHE_DATASET}")
    else:
        print(f"[t4-kit] cache dataset not found -> fresh build: {CACHE_DATASET}")

    src = KERNEL_SRC.read_text()
    src, n = re.subn(r"^CFG = None  # __LAUNCHER_CONFIG__.*$",
                     f"CFG = {cfg!r}", src, count=1, flags=re.M)
    if n != 1:
        sys.exit("serve_t4_gpu.py is missing the __LAUNCHER_CONFIG__ line")

    if args.dry:
        print(json.dumps(cfg, indent=2))
        print(f"\nwould push {KERNEL_ID} with dataset_sources={dataset_sources}")
        return

    RUNTIME.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "serve_t4_gpu.py").write_text(src)
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": KERNEL_ID,
            "title": KERNEL_SLUG,
            "code_file": "serve_t4_gpu.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": dataset_sources,
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=1))
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(td)],
                           capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        print(out[-1600:])
        if "successfully pushed" not in out:
            sys.exit("Push failed")

    STATE_FILE.write_text(json.dumps({
        "kernel": KERNEL_ID,
        "topic": cfg["ntfy_topic"],
        "api_key": cfg["api_key"],
        "keepalive_min": cfg["keepalive_min"],
    }))
    STATE_FILE.chmod(0o600)
    print(f"[t4-kit] state -> {STATE_FILE}")


if __name__ == "__main__":
    main()
