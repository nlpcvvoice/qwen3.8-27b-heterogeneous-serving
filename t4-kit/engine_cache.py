#!/usr/bin/env python3
"""Engine cache tools for the T4 kit.

Commands:
  cache.py validate <engine.tar.gz>          check tar structure (launcher + libs)
  cache.py upload   <engine.tar.gz> [--sha] [--msg]   push to private cache dataset
  cache.py fetch    [--out DIR] [--check]    download + sha/structure verify

The engine.tar.gz produced by a source-build run bundles BUILD/bin + BUILD/lib
(the thin llama-server launcher AND libllama-server-impl.so / libggml*.so).
A 17 KB thin shell alone is NOT cacheable. Pushes are free (no GPU quota).
Never prints the Kaggle token."""
import argparse, json, os, subprocess, sys, tarfile, tempfile
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kit_config import CACHE_DATASET, MIN_TAR_BYTES, RUNTIME
import kaggle_login as kl  # noqa: E402


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def validate_engine_tar(path: Path):
    """Return (ok, problems). Checks it is a gzip tar with llama-server + runtime libs."""
    probs = []
    try:
        with tarfile.open(path, "r:gz") as tf:
            names = tf.getnames()
    except Exception as exc:  # noqa: BLE001
        return False, [f"not a gzip tar: {exc.__class__.__name__}"]
    has_srv = any(n == "bin/llama-server" for n in names)
    has_libs = any((n.startswith("bin/") or n.startswith("lib/")) and
                   any(k in n for k in ("libllama-server-impl.so", "libggml")) for n in names)
    if not has_srv:
        probs.append("bin/llama-server missing")
    if not has_libs:
        probs.append("no libllama-server-impl.so / libggml*.so inside (thin shell?)")
    return (not probs), probs


def _dataset_exists(api) -> bool:
    try:
        api.dataset_status(CACHE_DATASET)
        return True
    except Exception:
        return False


def cmd_validate(path: Path) -> int:
    if path.stat().st_size < MIN_TAR_BYTES:
        print(f"too small (<{MIN_TAR_BYTES//1_000_000} MB) -> not a valid engine bundle")
        return 1
    ok, probs = validate_engine_tar(path)
    if not ok:
        print("INVALID: " + "; ".join(probs))
        return 1
    with tarfile.open(path, "r:gz") as tf:
        names = sorted(tf.getnames())
    impl = [n for n in names if "impl" in n or "ggml" in n]
    print(f"{path.name}: {path.stat().st_size:_} bytes")
    print(f"  bin/llama-server          present")
    print(f"  runtime libs ({len(impl)}): {impl[:6]}")
    print("OK")
    return 0


def cmd_upload(path: Path, sha: str, msg: str) -> int:
    if path.stat().st_size < MIN_TAR_BYTES:
        sys.exit(f"engine tarball too small (<{MIN_TAR_BYTES//1_000_000} MB)")
    ok, probs = validate_engine_tar(path)
    if not ok:
        sys.exit("INVALID: " + "; ".join(probs))
    kl.login()
    api = kl.get_kaggle_api()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=RUNTIME) as td:
        stage = Path(td)
        dst = stage / "engine.tar.gz"
        import shutil
        shutil.copy2(path, dst)
        sha = sha or sha256_file(dst)
        (stage / "manifest.json").write_text(
            f'{{"sha256":"{sha}","size":{dst.stat().st_size},"toolchain":"CUDA-75","artifact":"engine.tar.gz","gzip":true}}\n')
        (stage / "dataset-metadata.json").write_text(json.dumps({
            "id": CACHE_DATASET,
            "title": "llama-server (Qwen3.8-27B) engine cache",
            "subtitle": "prebuilt CUDA llama-server runtime bundle (bin+lib) to skip 25-min cmake build",
            "isPrivate": "true",
            "licenses": [{"name": "other"}],
        }, indent=1))
        if _dataset_exists(api):
            cmd = ["kaggle", "datasets", "version", "-p", str(stage), "-m", msg, "--dir-mode", "zip"]
        else:
            cmd = ["kaggle", "datasets", "create", "-p", str(stage), "--dir-mode", "zip"]
        r = subprocess.run([sys.executable, "-m", *cmd], capture_output=True, text=True)
        print(((r.stdout or "") + (r.stderr or ""))[-1200:])
        if r.returncode != 0:
            sys.exit("dataset push failed")
    print(f"[t4-kit] uploaded engine.tar.gz sha256={sha}")
    print(f"[t4-kit] {CACHE_DATASET} versioned (private)")
    return 0


def cmd_fetch(out: Path, check: bool) -> int:
    kl.login()
    api = kl.get_kaggle_api()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=RUNTIME) as td:
        r = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download",
                            "-d", CACHE_DATASET, "-p", td, "--dir-mode", "zip"],
                           capture_output=True, text=True)
        print(((r.stdout or "") + (r.stderr or ""))[-600:])
        if r.returncode != 0:
            sys.exit("dataset download failed")
        zips = list(Path(td).glob("*.zip"))
        if not zips:
            sys.exit("no zip in download")
        with ZipFile(zips[0]) as z:
            names = z.namelist()
            mani = [n for n in names if n.endswith("manifest.json")]
            expect = None
            if mani:
                expect = json.loads(z.read(mani[0]))["sha256"]
            tars = [n for n in names if n.endswith("engine.tar.gz")]
            if not tars:
                sys.exit("engine.tar.gz not in dataset (legacy cache; re-run upload)")
            out_tar = out / "engine.tar.gz"
            with z.open(tars[0]) as src, open(out_tar, "wb") as dst:
                os.write(dst.fileno(), src.read())
    local = sha256_file(out_tar)
    print(f"[t4-kit] engine.tar.gz -> {out_tar} ({out_tar.stat().st_size:_} bytes)")
    if expect:
        print(f"[t4-kit] sha256 {local}  (manifest {expect})")
        if expect != local:
            sys.exit("sha MISMATCH")
    else:
        print(f"[t4-kit] sha256 {local}  (no manifest in dataset)")
    with tarfile.open(out_tar, "r:gz") as tf:
        names = sorted(tf.getnames())
    impl = [n for n in names if "impl" in n or "ggml" in n]
    print(f"[t4-kit] bin/llama-server {'OK' if 'bin/llama-server' in names else 'MISSING'}")
    print(f"[t4-kit] runtime libs ({len(impl)}): {impl[:6]}")
    if not ("bin/llama-server" in names and impl):
        sys.exit("engine tar contents invalid")
    if check and expect is None:
        sys.exit("no manifest to verify against")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pv = sub.add_parser("validate", help="check an engine.tar.gz offline")
    pv.add_argument("tar", type=Path)
    pu = sub.add_parser("upload", help="push engine.tar.gz to private cache dataset")
    pu.add_argument("tar", type=Path)
    pu.add_argument("--sha", default="", help="expected sha256 (default: computed)")
    pu.add_argument("--msg", default="update-engine", help="dataset version message")
    pf = sub.add_parser("fetch", help="download + verify the cached engine")
    pf.add_argument("--out", type=Path, default=ROOT / "out")
    pf.add_argument("--check", action="store_true", help="require manifest sha match")
    args = ap.parse_args()

    rc = {"validate": cmd_validate, "upload": cmd_upload, "fetch": cmd_fetch}[args.cmd]
    if args.cmd == "validate":
        sys.exit(rc(args.tar))
    if args.cmd == "upload":
        sys.exit(rc(args.tar, args.sha, args.msg))
    sys.exit(rc(args.out, args.check))


if __name__ == "__main__":
    main()