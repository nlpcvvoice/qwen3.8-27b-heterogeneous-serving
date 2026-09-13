#!/usr/bin/env bash
# Build the engine factory image and produce engine.tar.gz into $OUT (default ./out).
# Usage: ./build_engine.sh [OUT_DIR] [--commit <sha-or-master>]
#   OUT_DIR defaults to ./out. Embed the base image override via DOCKER_BUILDKIT=1 too.
# Requires: docker (no GPU needed; nvcc compiles on CPU).
set -euo pipefail
OUT="$(realpath "${1:-./out}")"
shift || true
COMMIT=master
if [[ "${1:-}" == "--commit" ]]; then COMMIT="${2:?usage: --commit <sha|master>}"; fi
cd "$(dirname "$0")"
mkdir -p "$OUT"
TAG="qwen38-engine-factory"
echo ">> building image (commit=$COMMIT) ..."
docker build --build-arg LLAMA_COMMIT="$COMMIT" -t "$TAG" .
echo ">> packaging engine -> $OUT ..."
docker run --rm -v "$OUT":/out "$TAG"
echo ">> done: $OUT/engine.tar.gz  (sha: $(cat "$OUT/engine.sha"))"