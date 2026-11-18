#!/bin/bash
# waits for E120a, then: refined token files -> chain data on the refined graph -> consumers (E121a), keep evaluation, plug / fine-tune / joint arms (E121b)
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
mkdir -p $O/e121_a $O/e121_b $O/e121_b_keep
rm -f $O/e121.done
until [ -f $O/e120a.done ]; do sleep 20; done
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p15_make_tokens.py --name e120_a --out tok_e120_a > $O/e121_a/tokens.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p15_chain_data.py --out chain_r --tok tok_e120_a > $O/e121_a/data.log 2>&1
(bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p14_chain_consumer.py --data chain_r --out e121_a --modes route+ego,geo,route --seeds 3 > $O/e121_a/run.log 2>&1; echo done > $O/e121a.done) &
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p15_keep_eval.py --name e120_a > $O/e120_a/keep_eval.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p15_joint.py --arm plug --out e121_b > $O/e121_b/plug.log 2>&1
for arm in cons joint0 joint1; do
  (bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p15_joint.py --arm $arm --out e121_b --epochs 2 --workers 2 > $O/e121_b/$arm.log 2>&1) &
done
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p15_joint.py --arm joint1 --keep_in 1 --out e121_b_keep --epochs 2 --workers 2 > $O/e121_b_keep/joint1_keep.log 2>&1
wait
echo done > $O/e121.done
