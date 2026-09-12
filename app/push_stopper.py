#!/usr/bin/env python3
"""Push a tiny free CPU kernel that posts CONTROL-stop to the running serve kernel's ntfy topic,
letting it self-terminate (quit quota) when this box cannot reach ntfy.sh directly.
Writes nothing new; reads topic from tmp/kaggle-gpu-lab.json. Never prints tokens/topics fully."""
import json, re, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kaggle_login as kl

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "p3" / "kernel" / "ctrl_stop.py"
USER = "tentenshishi"
SLUG = "qwen38-gpu-ctrl"


def main() -> None:
    kl.login()
    state = json.loads((ROOT / "tmp" / "kaggle-gpu-lab.json").read_text())
    topic = state["topic"]
    print(f"target topic ...{topic[-6:]}")

    src = SRC.read_text()
    src, n = re.subn(r"^TOPIC = None  # __STOPPER_TOPIC__.*$",
                     f"TOPIC = {topic!r}", src, count=1, flags=re.M)
    if n != 1:
        sys.exit("ctrl_stop.py is missing the __STOPPER_TOPIC__ line")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "ctrl_stop.py").write_text(src)
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{USER}/{SLUG}",
            "title": SLUG,
            "code_file": "ctrl_stop.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "false",
            "enable_internet": "true",
            "machine_shape": "CPU",
            "competition_sources": [], "dataset_sources": [],
            "kernel_sources": [], "model_sources": [],
        }, indent=1))
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(td)],
                           capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        print(out[-1200:])
        if "successfully pushed" not in out:
            sys.exit("Push failed")


if __name__ == "__main__":
    main()