#!/usr/bin/env bash
# Rasterizer-Quelle fuer neuere Compiler: fehlendes #include <cstdint> in den Kopfdateien ergaenzen (uint64_t/uint32_t)
cd "$HOME/work/gaussiancar/gaussiancar/ops/diff-gaussian-rasterization/cuda_rasterizer" || exit 1
for f in rasterizer_impl.h auxiliary.h forward.h backward.h rasterizer.h config.h; do
  if [ -f "$f" ] && ! grep -q "<cstdint>" "$f"; then
    if grep -q "#pragma once" "$f"; then sed -i '0,/#pragma once/s//#pragma once\n#include <cstdint>/' "$f"; else sed -i '1i #include <cstdint>' "$f"; fi
    echo "ergaenzt: $f"
  fi
done
grep -n "cstdint" *.h
