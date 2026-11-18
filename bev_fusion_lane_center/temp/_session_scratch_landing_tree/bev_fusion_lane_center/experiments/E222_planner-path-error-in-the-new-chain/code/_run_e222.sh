#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E222: planner-level effect (consumer path error at 20 m, failure half) of fork pruning and of the failure-trained route head in the best chain (map builder B + learned gate, exact map); three consumer seeds
R=$BFLC
X=$R/experiment_execution/runs
until [ -f $X/E220.a/done ]; do sleep 30; done
A0="cov_r=0.5 vtx_r=1.5 junc_thr=0.5 reloc_thr=0.4 reloc_shift=12 reloc_tol=1.6"
S=dark_gateB,blur_gateB,cam_front3_off_gateB
python3 $R/tools/experiment/run.py --id E222.b --gpu-max-mb 9000 --retries 1 --basis E212,E215,E214 --seed 0 \
  --hyp "fork pruning and the failure-trained route head lower the planner path error under failure in the best chain" \
  --pred "path error at 20 m (night, failure half) falls from the loop-17 gate + exact map value (0.28 m) or stays within 0.05 m; blur / three cameras off likewise" \
  --rule "report; improvement counts only if the three-seed means differ by more than the seed spread" -- bash -c "
  bash $R/tools/env/_py14.sh p17_consumer.py --out e18_cons --series $S --mapnet 1 --ckpt e17_map_b --map_mode exact --route p16_f2hr_route_s0 --asm $A0 --tag_suffix _s0_noprune &&
  bash $R/tools/env/_py14.sh p17_consumer.py --out e18_cons --series $S --mapnet 1 --ckpt e17_map_b --map_mode exact --route p16_f2hr_route_s0 --asm $A0 fork_prune=0.3 --tag_suffix _s0_prune &&
  bash $R/tools/env/_py14.sh p17_consumer.py --out e18_cons --series $S --mapnet 1 --ckpt e17_map_b --map_mode exact --route e18_joint_c0 --asm $A0 fork_prune=0.3 --tag_suffix _c0_prune" > $R/runs/e17/logs/e222_a.log 2>&1
