#!/usr/bin/env python3
"""Upload tmp/gguf_private as a private Kaggle dataset."""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kaggle_login as kl
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "tmp" / "gguf_private"
SLUG = "qwen3-8-27b-q4-k-m-private"

kl.login()
api = kl.get_kaggle_api()
meta = json.loads((RELEASE / "dataset-metadata.json").read_text())
print("uploading:", RELEASE, "->", meta["id"])

t0 = time.time()
try:
    resp = api.dataset_create_new(str(RELEASE), public=False, quiet=True, convert_to_csv=False)
    print("dataset_create_new OK, waited %.1fs -> %s" % (time.time() - t0, resp))
except Exception as e:
    print("create_new err:", repr(e)[:300])
    try:
        resp2 = api.dataset_create_version(str(RELEASE), version_notes="update",
                                           quiet=True, convert_to_csv=False)
        print("create_version OK, waited %.1fs -> %s" % (time.time() - t0, resp2))
    except Exception as e2:
        print("create_version err:", repr(e2)[:300])
        sys.exit(1)
print("DONE")