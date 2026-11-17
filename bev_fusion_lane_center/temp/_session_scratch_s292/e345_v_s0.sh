#!/bin/bash
# E345 stage e324s_6, seed 0 only (its launcher was killed at 11:25 to free the GPU queue for the E351 timing block; seed 1 finished ok, 986 s):
# the same run.py job as day21.sh's s6 for E345.v_e324s_s0 (hyp / pred / rule frozen in the run record), then the stage marker and a restart of the chain.
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
python3 ../autoresearch/tools/run.py --id E345.v_e324s_s0 --seed 0 --gpu --retries 0 --require-card --split val --timeout 720 \
  --hyp "D99 step 0 (E345 part A): how much of each unit of the stored junction system can be taken away without retraining; job E345.v_e324s_s0" \
  --pred "E345 card section 3: v (all FREE head units) loses 0.005 topo F1 on the judgement half and saves 0.8 GMACs of the heads' 3.278; at least 15 of 45 units have a sweep-FREE point" \
  --rule "E345 card sections 0 and 3, by hand: analysis, selected on half of the VAL scenes and judged on the other half; loss = null minus ablated on the worse of the snapped and the differentiable read-out" \
  -- bash tools/env/_pyr.sh jprune_probe.py sweep --run e324_s_s0 --out temp/e345/e324_s_s0 --score-points temp/e345/e324_s_s0/score_points.json
rc=$?
echo "E345.v_e324s_s0 exit $rc $(date +%H:%M:%S)"
if [ $rc -eq 0 ]; then
  touch temp/day21/e324s_6.done
  rm -f temp/day21/FAIL_e324s_6
  echo "day21 restarted after stage 6 $(date)" >> temp/day21/day21.log
  bash temp/day21/day21.sh > temp/day21/day21_stdout4.log 2>&1
  echo "day21 exit $? $(date +%H:%M:%S)"
fi
