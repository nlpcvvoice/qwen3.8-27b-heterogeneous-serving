#!/usr/bin/env bash
# t4-kit one-click entry. T4 (GPU) only — no TPU.
#   ./start.sh test            offline self-test (zero quota)
#   ./start.sh check           live: token/quota/kernel/datasets
#   ./start.sh push [flags]    push the T4 kernel (see push_job.py --help)
#   ./start.sh watch [flags]   watch phases; --stop-on built auto-stops on engine-cached
#   ./start.sh stop            send CONTROL-stop (frees the GPU quota)
#   ./start.sh download        download kernel output to <project>/out
#   ./start.sh cache-validate <tar>
#   ./start.sh cache-upload <tar> [--sha ...]
#   ./start.sh cache-fetch [--check]
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PY=""
for c in "$DIR/Venv/bin/python" "$DIR/../Venv/bin/python" python3; do
  if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "no python3 found"; exit 1; }

CMD="${1:-test}"
shift || true
case "$CMD" in
  test)            exec "$PY" self_test.py "$@" ;;
  check)           exec "$PY" check.py "$@" ;;
  push)            exec "$PY" push_job.py "$@" ;;
  watch)           exec "$PY" watch_job.py "$@" ;;
  stop)            exec "$PY" stop_job.py "$@" ;;
  download)        exec "$PY" download_output.py "$@" ;;
  cache-validate)  exec "$PY" engine_cache.py validate "$@" ;;
  cache-upload)    exec "$PY" engine_cache.py upload "$@" ;;
  cache-fetch)     exec "$PY" engine_cache.py fetch "$@" ;;
  *) echo "unknown cmd: $CMD (test|check|push|watch|stop|download|cache-*)"; exit 1 ;;
esac