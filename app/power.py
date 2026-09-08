#!/usr/bin/env python3
"""One-key control for the whole serving stack.

  ./Venv/bin/python app/power.py status         -> controller + router state
  ./Venv/bin/python app/power.py on             -> power on (scheduler active)
  ./Venv/bin/python app/power.py off            -> kill switch: both engines stop
  ./Venv/bin/python app/power.py live           -> switch controller to live mode
  ./Venv/bin/python app/power.py dry            -> switch controller to dry mode
  ./Venv/bin/python app/power.py start gpu|tpu  -> manual engine start (live only)
"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8091"


def call(path, method="GET", body=None):
    req = urllib.request.Request(f"{BASE}{path}",
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"},
                                 method=method)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "help":
        print(__doc__); return
    cmd = sys.argv[1]
    if cmd == "status":
        st = call("/admin/status")
        print(f"power={st['power']} mode={st['mode']} primary={st['primary']} ({st['reason']})")
        for k in ("tpu", "gpu"):
            e = st[k]
            print(f"  {k}: {e['status']} healthy={e['healthy']} endpoint={e['endpoint']}")
    elif cmd in ("on", "off"):
        print(call("/admin/power", "POST", {"power": cmd == "on"}))
    elif cmd in ("live", "dry"):
        print(call("/admin/mode", "POST", {"mode": cmd}))
    elif cmd == "start" and len(sys.argv) == 3:
        print(call("/admin/action", "POST", {"act": f"start-{sys.argv[2]}"}))
    elif cmd == "stop" and len(sys.argv) == 3:
        print(call("/admin/action", "POST", {"act": f"stop-{sys.argv[2]}"}))
    elif cmd == "sim" and len(sys.argv) == 4 and sys.argv[3] in ("on", "off"):
        key = sys.argv[2]
        print(call("/admin/sim", "POST", {key: sys.argv[3] == "on"}))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()