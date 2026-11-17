#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
O=temp/_session_scratch_e347
D=$O/score_old_new/e332_ma_s0
export E347_FAST=1
run() { echo "=== $*"; bash temp/e347/run_staged.sh "$@" 2>&1 | grep -v "^run_staged: stage" | tail -${TAILN:-6} | cut -c1-330; echo "exit ${PIPESTATUS[0]}"; }
TAILN=10 run grid/test_jmaps_r2.py
TAILN=3 run grid/test_jmaps_simplify.py
TAILN=1 run jmaps_eval.py selftest
echo "=== --ckpt on the scratch copy of e332_ma_s0 (ckpt.pt = step 3000)"
export CUDA_VISIBLE_DEVICES=""
bash temp/e347/run_staged.sh evaluation/jmaps_eval.py score --run ../$O/score_old_new/e332_ma_s0 --ckpt --max 50 --workers 0 --base /mnt/c/Auto-research/bev_fusion_lane_center/runs/e273_p_s0/model.pt 2>&1 | grep -E "ckpt|written|gate|EDGE|Error|error" | cut -c1-320
ls $D
python3 - <<'PY'
import json
r = json.load(open("/mnt/c/Auto-research/bev_fusion_lane_center/temp/_session_scratch_e347/score_old_new/e332_ma_s0/jmaps_val_ckpt.json"))
print("ckpt_step", r["ckpt_step"], "frames", r["frames"], "kill-2 read: kp_ours_tol3.0_noang_ent pred =", r["keypoints"]["kp_ours_tol3.0_noang_ent"]["pred"])
PY
unset CUDA_VISIBLE_DEVICES
echo "=== mutations (round 2) -> $O/mut_round2.log"
python3 temp/e347/mutate.py m30_off_near_1m m31_offset_to_first_end m32_loss_mask_is_onehot m33_offset_per_kind_norm m34_gate_grid_coarse m35_edge_never m36_readouts_swapped m37_readout_names_swapped m38_best_by_f1_1_5 m39_dup_boundary m40_ckpt_wrong_key m43_ckpt_weights_only m41_sigma_guard_off m42_names_without_joffm m01_last_end_wins m05_end_keys_swapped m06_offset_mean_all_cells m07_offset_unmasked m20_dup_counts_matched m24_offset_term_off > $O/mut_round2.log 2>&1
echo "=== chain9 done"
