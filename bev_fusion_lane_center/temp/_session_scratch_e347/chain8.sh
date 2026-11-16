#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
O=temp/_session_scratch_e347
D=$O/score_old_new/e332_ma_s0
bash temp/e347/run_staged.sh jmaps.py > /dev/null 2>&1   # a full call (bflc_env.sh) refreshes the PYTHONPATH cache
export E347_FAST=1
run() { echo "=== $*"; bash temp/e347/run_staged.sh "$@" 2>&1 | grep -v "^run_staged: stage" | tail -${TAILN:-6} | cut -c1-330; echo "exit ${PIPESTATUS[0]}"; }
TAILN=12 run grid/test_jmaps_r2.py
TAILN=1 run jmaps_eval.py selftest
TAILN=1 run jend_grid_stats.py selftest
TAILN=2 run grid/test_jmaps_simplify.py
TAILN=1 run jgt.py
TAILN=1 run qjunction.py
TAILN=1 run jmaps.py
TAILN=1 run jends.py
TAILN=1 run jmodel_cost.py selftest
TAILN=1 run grid/test_jmaps_dirx.py
TAILN=1 run jends_eval.py selftest
echo "=== scorer: new vs the old one (24 VAL frames, stored e332_ma_s0, a scratch copy)"
export CUDA_VISIBLE_DEVICES=""
rm -f $D/jmaps_val.json
bash temp/e347/run_staged.sh evaluation/jmaps_eval.py score --run ../$O/score_old_new/e332_ma_s0 --max 24 --workers 0 --base /mnt/c/Auto-research/bev_fusion_lane_center/runs/e273_p_s0/model.pt > $D/new2.log 2>&1 && mv $D/jmaps_val.json $D/new2.json
tail -n 12 $D/new2.log | grep -E "gate|EDGE|written" | cut -c1-300
python3 $O/cmp_old_new.py new2
echo "=== --ckpt on the same copy (the stored ckpt.pt of e332_ma_s0, step 3000)"
bash temp/e347/run_staged.sh evaluation/jmaps_eval.py score --run ../$O/score_old_new/e332_ma_s0 --ckpt --max 8 --workers 0 --base /mnt/c/Auto-research/bev_fusion_lane_center/runs/e273_p_s0/model.pt 2>&1 | grep -E "ckpt|written|gate" | cut -c1-300
ls $D
echo "=== trainers smoke (quick)"
bash temp/e347/run_staged.sh smoke_trainers.py --quick 2>&1 | grep -v "^run_staged" | cut -c1-260 | tail -12
TAILN=4 run grid/test_jmaps_r2.py --real --only none
echo "=== mutations (round 2)"
python3 temp/e347/mutate.py m30_off_near_1m m31_offset_to_first_end m32_loss_mask_is_onehot m33_offset_per_kind_norm m34_gate_grid_coarse m35_edge_never m36_readouts_swapped m37_readout_names_swapped m38_best_by_f1_1_5 m39_dup_boundary m40_ckpt_wrong_key m41_sigma_guard_off m42_names_without_joffm m01_last_end_wins m05_end_keys_swapped m06_offset_mean_all_cells m07_offset_unmasked m20_dup_counts_matched m24_offset_term_off 2>&1 | cut -c1-260
echo "=== chain8 done"
