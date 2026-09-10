#!/usr/bin/env python3
"""P3: report current engine-cache dataset status (priv/versions/size/sha).

Read-only; never pushes anything. Uses the Kaggle API (masked creds only).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_login as kl  # noqa: E402
from common import CACHE_DATASET  # noqa: E402


def main() -> None:
    kl.login()
    api = kl.get_kaggle_api()
    try:
        md = api.dataset_status(CACHE_DATASET)
    except Exception as exc:
        print(f"[P3] cache dataset NOT available: {type(exc).__name__}")
        sys.exit(1)

    print(f"[P3] dataset : {CACHE_DATASET}")
    print(f"[P3] title   : {md.get('title', '')}")
    print(f"[P3] private : {md.get('isPrivate')}")
    print(f"[P3] files   : {len(md.get('files', []))}")
    total = sum(f.get('totalBytes', 0) for f in md.get('files', []))
    print(f"[P3] total   : {total/1e6:.1f} MB")
    for f in md.get('files', []):
        print(f"[P3]   - {f.get('name')}  {f.get('totalBytes', 0)/1e6:.1f} MB")
    print(f"[P3] currentVersionNumber: {md.get('currentVersionNumber', '?')}")


if __name__ == "__main__":
    main()