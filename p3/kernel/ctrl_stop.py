#!/usr/bin/env python3
"""CPU (free) remote kill-switch kernel: posts CONTROL-stop to the target ntfy topic.
Runs on Kaggle's network which can reach ntfy.sh even when this box cannot.
Inject TOPIC via __STOPPER_TOPIC__ before push."""
import json, time, urllib.request

TOPIC = None  # __STOPPER_TOPIC__

BODY = json.dumps({
    "topic": TOPIC,
    "title": "ctrl stop request",
    "message": json.dumps({"phase": "CONTROL", "cmd": "stop"}),
})


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def post() -> bool:
    try:
        req = urllib.request.Request(
            "https://ntfy.sh",
            data=BODY.encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            ok = r.status == 200
        log(f"post OK ({r.status})" if ok else f"post HTTP {r.status}")
        return ok
    except Exception as e:
        log(f"post failed: {type(e).__name__}: {e}")
        return False


def main() -> None:
    log(f"stopper online, topic={TOPIC}")
    ok = 0
    for i in range(16):  # ~4 min of blanket posts
        if post():
            ok += 1
        time.sleep(15)
    log(f"done, {ok}/16 posts ok")


if __name__ == "__main__":
    main()