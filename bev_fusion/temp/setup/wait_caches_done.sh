#!/usr/bin/env bash
# One-off: wait until both cache builds (feature cache, GT prefill) report "all done" in their logs; prints one line.
cd "$(dirname "$0")/../.." || exit 1
until grep -q "all done" temp/checks/feature_cache_build.log 2>/dev/null && grep -q "all done" temp/checks/gt_cache_build.log 2>/dev/null; do
  sleep 60
done
echo "both caches done $(date +%T)"
