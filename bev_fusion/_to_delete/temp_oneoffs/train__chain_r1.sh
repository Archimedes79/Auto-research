#!/usr/bin/env bash
# One-off: when t1_r0 has finished (.done, or its process is gone), start both round-1 arms at once from the pinned code
# 2ddf511 (queued jobs of other chats start as soon as the GPU looks free, so there must be no gap).
cd "$(dirname "$0")/../.." || exit 1
until [ -f runs/t1_r0/.done ] || ! pgrep -f "[t]1_r0.json" > /dev/null; do sleep 10; done
echo "t1_r0 finished or gone $(date +%T); .done: $([ -f runs/t1_r0/.done ] && echo yes || echo NO)"
bash temp/train/run_cfg.sh training/configs/t1_r1a_dir.json 2ddf511 &
sleep 20
bash temp/train/run_cfg.sh training/configs/t1_r1b_uwso.json 2ddf511 &
wait
echo "round 1 finished $(date +%T)"
