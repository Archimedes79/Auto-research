#!/usr/bin/env bash
# One-off (t2_block_swaps): C: is full (8.8 GB free, 14 runs x 0.9 GB). Moves the checkpoints of every FINISHED and EVALUATED
# study run (eval/metrics.json exists) to D: (nothing is deleted: the copy is size-checked before the original goes), leaves a
# MOVED.txt in the run folder. Re-evaluation needs the file back: cp <D: folder>/ckpt_*.pt runs/t2_<arm>/.
# Usage: bash temp/study/move_ckpts.sh [--loop]   (loop: repeat every 10 min until temp/study/lane.log says "lane finished")
cd "$(dirname "$0")/../.." || exit 1
DEST=/mnt/d/Auto-research/bev_fusion_ckpts
move_once() {
  for r in runs/t2_*; do
    [ -f $r/eval/metrics.json ] && [ -f $r/.done ] || continue
    ls $r/ckpt_*.pt >/dev/null 2>&1 || continue
    a=$(basename $r); mkdir -p $DEST/$a
    for f in $r/ckpt_*.pt; do
      cp "$f" $DEST/$a/ && [ "$(stat -c %s "$f")" = "$(stat -c %s $DEST/$a/$(basename $f))" ] && rm "$f" \
        && echo "$(date +%T) moved $f -> $DEST/$a/"
    done
    echo "checkpoints moved to D: $DEST/$a/ on $(date +%F) (C: was full); copy back to re-evaluate" > $r/MOVED.txt
  done
}
move_once
if [ "$1" = "--loop" ]; then
  until grep -q "lane finished" temp/study/lane.log; do sleep 600; move_once; done
  move_once
fi
