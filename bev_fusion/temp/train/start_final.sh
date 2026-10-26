#!/usr/bin/env bash
# One-off: wait for t1_r0 to finish (its process is gone), then start the final combined training from the pinned commit.
# Usage: bash temp/train/start_final.sh <config.json> <commit>
cd "$(dirname "$0")/../.." || exit 1
until ! pgrep -f "[t]1_r0.json" > /dev/null; do sleep 5; done
echo "t1_r0 gone $(date +%T); .done: $([ -f runs/t1_r0/.done ] && echo yes || echo NO)"
bash temp/train/run_cfg.sh "$1" "$2"
