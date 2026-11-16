#!/bin/bash
# E347 ledger notes (numbers from runs/e347_<arm>_s<S>/jmaps_val.json and runs/e332_ma_s<S>/jmaps_val_legacy.json; card section 3)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
L="python3 ../autoresearch/tools/ledger.py note"
DEC="by hand: maps scorer (stage 1 label and kill gate); both arms FAIL on both seeds -> E347 closes at stage 1 (the ends need the 1 m grid for this recipe), no stage 2"
$L E347.g_s0 --result "3000 steps ok; scored (E347.score_g_s0): own rule (3 x 3 peak) best thr 0.55: F1@3 0.7084 vs control 0.7570 (-0.0486), F1@1.5 0.5643 (-0.0182), crowded entry recall 0.6013 (-0.1701)" --decision "FAIL clause on seed 0 ($DEC)" --notes "staged code temp/_session_scratch_e347/stage; kill 2 at step 500 passed (479 entry keypoints at 0.3)"
$L E347.g_s1 --result "3000 steps ok; scored: own rule best thr 0.55 / 0.60: F1@3 0.6996 vs 0.7505 (-0.0510), F1@1.5 0.5614 (-0.0191), crowded entry recall 0.6086 (-0.1857)" --decision "FAIL clause on seed 1 ($DEC)" --notes "kill 2 at step 500 passed (predictions present at the low thresholds)"
$L E347.o_s0 --result "3000 steps ok; scored: own rule (threshold alone) best thr 0.45: F1@3 0.6301 vs 0.7570 (-0.1269), F1@1.5 0.5278 (-0.0547), crowded 0.7172 (-0.0541), surplus-duplicate share 0.1843 (control 0.0010); with the peak step 0.6885" --decision "FAIL clause on seed 0 ($DEC)" --notes "kill 2 at step 500 passed (1505 entry keypoints at 0.3: the focal head fires broadly early)"
$L E347.o_s1 --result "3000 steps ok; scored: own rule best thr 0.45 / 0.40: F1@3 0.6253 vs 0.7505 (-0.1253), F1@1.5 0.5288 (-0.0517), crowded 0.6968 (-0.0975), duplicate share 0.2479; with the peak step 0.6911" --decision "FAIL clause on seed 1 ($DEC)" --notes "kill 2 at step 500 passed"
for A in g_s0 g_s1 o_s0 o_s1; do
  $L E347.kill2_$A --result "step 500: the checkpoint has predictions (alive); the kill criterion is not met" --decision "diagnostic, no verdict" --notes "jmaps_eval.py score --ckpt --max 50 (56 frames) on the staged scorer; the 0.3 row was read for g_s0 and o_s0"
  $L E347.score_$A --result "see E347.$A" --decision "stage 1 label by hand: maps scorer" --notes "the staged scorer on the finished run, default raster; card section 3"
done
$L E347.c_score_s0 --result "control (E332 arm a seed 0) with the E347 scorer: own rule best thr 0.80 / 0.75: F1@3 0.7570, F1@1.5 0.5826, crowded entry 0.7713, duplicates 0.0010; at 0.3: 0.7409 / 0.5631 / 0.7854 (reproduces the stored numbers)" --decision "control numbers written into the card before any arm read-out" --notes "runs/e332_ma_s0/jmaps_val_legacy.json, raster legacy, grid 0.05 - 0.95"
$L E347.c_score_s1 --result "control seed 1: best thr 0.70 / 0.75: F1@3 0.7505, F1@1.5 0.5805, crowded entry 0.7943, duplicates 0.0009; at 0.3: 0.7356 / 0.5657 / 0.8010 (reproduces the stored numbers)" --decision "control numbers written into the card before any arm read-out" --notes "runs/e332_ma_s1/jmaps_val_legacy.json"
echo "ledger notes done"
