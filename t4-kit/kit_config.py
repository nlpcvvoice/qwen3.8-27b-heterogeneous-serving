"""t4-kit shared constants (T4 / GPU only — no TPU)."""

import os
from pathlib import Path

KIT = Path(__file__).resolve().parent
PARENT_DIR = KIT.parent  # project root that carries the kit (deploy target)

# ---- the Kaggle account (single-user kit) -------------------------------
T4_OWNER = "tentenshishi"
KERNEL_ID = f"{T4_OWNER}/qwen38-t4-serve"
KERNEL_SLUG = "qwen38-t4-serve"
WEIGHTS_DATASET = f"{T4_OWNER}/qwen3-8-27b-q4-k-m-private"  # 16.46 GB GGUF mirror
CACHE_DATASET = f"{T4_OWNER}/llama-server-qwen38-cache"      # engine.tar.gz cache
SERVED_MODEL_NAME = "qwen3.8-27b"
MACHINE_SHAPE = "NvidiaTeslaT4"  # 2x T4, 16 GB each

# ---- runtime state (never committed; MASTER: all tmp files under ./tmp) --
RUNTIME = Path(os.environ.get("T4KIT_RUNTIME_DIR") or (PARENT_DIR / "tmp" / "t4-kit"))
STATE_FILE = RUNTIME / "t4-job.json"       # {kernel, topic, api_key, keepalive_min}
CFG_DIR = RUNTIME / "kaggle_cfg"           # KAGGLE_CONFIG_DIR (keeps writes in-project)
WATCH_LOG = RUNTIME / "watch.log"
STAGE_DIR = RUNTIME / "stage"              # temp upload staging
OUT_DIR = PARENT_DIR / "out"               # downloaded kernel output root

# ---- token file (in-memory only; never printed) -------------------------
_env_tok = os.environ.get("KAGGLE_TOKEN_FILE", "").strip()
TOKEN_CANDIDATES: list[Path] = []
if _env_tok:
    TOKEN_CANDIDATES.append(Path(_env_tok))
TOKEN_CANDIDATES += [KIT / "reference" / "API-Token", PARENT_DIR / "reference" / "API-Token"]

# ---- sanity floors ------------------------------------------------------
MIN_LEGACY_BYTES = 1_000_000    # legacy single llama-server file (>1 MB)
MIN_TAR_BYTES = 20_000_000      # engine.tar.gz bundle (bin + lib, >20 MB)

KERNEL_SRC = KIT / "kernel" / "serve_t4_gpu.py"