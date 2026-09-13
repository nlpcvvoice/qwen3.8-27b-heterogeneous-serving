#!/usr/bin/env bash
# Write /out/engine.tar.gz (bin/ + lib/ layout, identical to the kernel's package_engine)
# and /out/engine.sha (resolved llama.cpp commit for versioning).
# Static-buid engines ship only bin/llama-server; lib/ is included when present.
set -euo pipefail
sha="$(git -C /src rev-parse HEAD)"
mkdir -p /out
dirs="bin"
[ -d /src/build/lib ] && dirs="bin lib"
tar -czf /out/engine.tar.gz -C /src/build $dirs
[ -s /out/engine.tar.gz ] && [ -x /src/build/bin/llama-server ]
printf '%s\n' "$sha" > /out/engine.sha
echo "engine.sha=$sha"
echo "tar entries: bin, lib(present: $([ -d /src/build/lib ] && echo yes || echo no))"
echo "engine.tar.gz size=$(du -h /out/engine.tar.gz | cut -f1)"
echo "NEEDED libs:"
readelf -d /src/build/bin/llama-server | grep -E 'libcuda|libcudart|libcublas|libllama' || true