#!/bin/bash
# final pass in the final state of the stage: full PYTHONPATH (no FAST), one test at a time
cd /mnt/c/Auto-research/bev_fusion_lane_center
O=temp/_session_scratch_e347
run() { echo "=== $*"; bash temp/e347/run_staged.sh "$@" 2>&1 | grep -v "^run_staged: stage" | tail -${TAILN:-6} | cut -c1-330; echo "exit ${PIPESTATUS[0]}"; }

bash temp/e347/run_staged.sh jmaps.py > /dev/null 2>&1   # a full call (bflc_env.sh) refreshes the PYTHONPATH cache
export E347_FAST=1
TAILN=3 run grid/test_jmaps_r2.py
TAILN=4 run grid/test_jmaps_simplify.py
TAILN=1 run grid/test_jmaps_dirx.py
TAILN=1 run jmaps.py
TAILN=1 run jends.py
TAILN=1 run jgt.py
TAILN=1 run qjunction.py
TAILN=1 run jmaps_eval.py selftest
TAILN=1 run jmodel_cost.py selftest
TAILN=1 run jend_grid_stats.py selftest
TAILN=2 run jends_eval.py selftest
echo "=== cost table of the 2 m head"
bash temp/e347/run_staged.sh jmodel_cost.py --ctx dil+nopaint+r2 2>&1 | grep -v "^run_staged"
echo "=== cost table of the E341 head (no paint, 1 m) and the E312 head"
bash temp/e347/run_staged.sh jmodel_cost.py --ctx dil+nopaint 2>&1 | grep -v "^run_staged" | head -16
echo "=== trainers smoke"
bash temp/e347/run_staged.sh smoke_trainers.py 2>&1 | grep -v "^run_staged" | cut -c1-330 | tail -30
echo "=== E344 scratch vs the new tool (same split)"
bash temp/e347/run_staged.sh temp/_session_scratch_s292/gt_end_grid.py 2>&1 | grep -v "^run_staged" | cut -c1-500 | tail -5
bash temp/e347/run_staged.sh jend_grid_stats.py --out $O/jend_grid_stats_val.json 2>&1 | grep -v "^run_staged" | cut -c1-500
echo "=== chain3 done"
