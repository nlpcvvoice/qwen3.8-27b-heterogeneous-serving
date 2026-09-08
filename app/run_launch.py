#!/usr/bin/env python3
"""Wrap kaggle-tpu-lab/launch.py so it inherits the in-memory Kaggle auth.

Sets KAGGLE_API_TOKEN / KAGGLE_USERNAME / KAGGLE_CONFIG_DIR in the child env,
then runs `python launch.py <args...>` from the repo directory. Never prints the
token (see MASTER.md: in-memory only).
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kaggle_login as kl

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "kaggle-tpu-lab"

kl.login()  # reads token in memory + exports KAGGLE_API_TOKEN / KAGGLE_CONFIG_DIR

os.environ["KAGGLE_USERNAME"] = "tentenshishi"  # avoids CLI username detection path

cmd = [sys.executable, "launch.py", *sys.argv[1:]]
print("running:", " ".join(cmd))
r = subprocess.run(cmd, cwd=str(REPO), env=os.environ)
sys.exit(r.returncode)
