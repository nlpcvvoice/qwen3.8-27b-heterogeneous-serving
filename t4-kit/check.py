#!/usr/bin/env python3
"""Local diagnostics: token verify, weekly GPU quota, kernel status, datasets."""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kit_config import CACHE_DATASET, KERNEL_ID, WEIGHTS_DATASET
import kaggle_login as kl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    kl.login()
    api = kl.get_kaggle_api()
    v = kl.verify(api)
    print(f"token : ok={v['ok']}  masked={v['token_masked']}  api={v['api']}  ({v['status']})")

    q = kl.quota()
    g = q["gpu"]
    if g is None:
        print("quota : gpu n/a")
    else:
        print(f"quota : GPU used {g['used_h']:.2f}h | remaining {g['remaining_h']:.2f}h "
              f"| total {g['total_h']:.2f}h  (refresh {q['refresh_at']})")

    if args.quiet:
        return
    print(f"kernel: {KERNEL_ID} ->", end=" ")
    try:
        kd = api.kernels_status(KERNEL_ID)
        print(getattr(kd, "status", "?"))
    except Exception as e:
        print(f"err:{type(e).__name__} (not pushed yet?)")

    for ref, tag in ((WEIGHTS_DATASET, "weights"), (CACHE_DATASET, "cache")):
        try:
            md = api.dataset_status(ref)
            print(f"ds    : {tag:7s} {ref}  -> {str(md)[:80]}")
        except Exception:
            print(f"ds    : {tag:7s} {ref}  MISSING")

    if not v["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()