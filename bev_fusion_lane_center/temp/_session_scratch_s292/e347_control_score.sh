#!/bin/bash
# E347: the control's extended numbers (card section 2b (e)): the maps scorer of the staged tree on the stored E332 arm a runs (e332_ma_s0 / s1), `--raster legacy`
# (the control was trained on the legacy raster; its result goes to runs/e332_ma_s<S>/jmaps_val_legacy.json, the stored jmaps_val.json is not touched), CPU, one seed after the other.
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
for S in 0 1; do
  python3 ../autoresearch/tools/run.py --id E347.c_score_s$S --seed $S --retries 0 --split val --timeout 240 \
    --hyp "E347 control: the 1 m maps head without the paint branch (E332 arm a, seed $S) scored with the E347 scorer (3 m and 1.5 m keypoint F1, crowded entry recall and surplus duplicates at its best threshold on the 0.05 - 0.95 grid) so that arms and control are read the same way" \
    --pred "the stored numbers come back unchanged at 0.3 (keypoint F1 at 3 m 0.7409 / 0.7355); at the best threshold 0.7561 / 0.7501 or slightly higher on the finer grid" \
    --rule "E347 card section 0 stage 1: numbers for the gate clauses, by hand: maps scorer (a label and a kill gate, not the verdict)" \
    -- bash temp/e347/run_staged.sh jmaps_eval.py score --run e332_ma_s$S --raster legacy --workers 2
  echo "E347.c_score_s$S exit $? $(date +%H:%M:%S)"
done
echo "control scoring done"
