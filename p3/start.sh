#!/usr/bin/env bash
# P3 engine-cache: one-click entry for ANY machine/agent.
#   ./start.sh test        -> full LOCAL validation (zero quota, fastest)
#   ./start.sh upload BIN  -> push an engine bundle (engine.tar.gz) to the private cache
#   ./start.sh fetch       -> download + sha-check the cached engine
#   ./start.sh status      -> show dataset state
#   ./start.sh live-test   -> real Kaggle roundtrip (needs quota, optional)
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PY=""
for c in "$DIR/Venv/bin/python" "$DIR/../Venv/bin/python" python3; do
  if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then echo "no python3 found"; exit 1; fi

CMD="${1:-test}"
case "$CMD" in
  test)        exec "$PY" validate_cache.py ;;
  static)      exec "$PY" validate_cache.py --static-only ;;
  live-test)   exec "$PY" validate_cache.py --live ;;
  status)      exec "$PY" status.py ;;
  upload)      [ $# -ge 2 ] || { echo "usage: ./start.sh upload <engine.tar.gz>"; exit 1; }
               exec "$PY" upload_cache.py "$2" ;;
  fetch)       exec "$PY" fetch_cache.py --check ;;
  *) echo "unknown cmd: $CMD (test|static|live-test|status|upload|fetch)"; exit 1 ;;
esac