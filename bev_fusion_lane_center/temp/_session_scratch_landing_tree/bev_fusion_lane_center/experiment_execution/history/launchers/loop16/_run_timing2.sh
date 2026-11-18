#!/bin/bash
# second timing window (idle GPU) after the point-spacing trainings: the assembly became faster (shared maximum filters, vectorised relocation), the 2 m / 8 m route heads exist for the route-head timing
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/diag_resample
T=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/timing
rm -f $T/done2
while [ ! -f $O/done_steps ]; do sleep 30; done
sleep 20
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_pareto.py --n 60 > $T/pareto2.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_pipeline.py --name e123_D7_r3_f1 --n 60 > $T/pipeline_f1_2.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_pipeline.py --name e123_D7_r3_j1 --n 60 > $T/pipeline_j1_2.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_pipeline.py --name e123_D7_r3_j1 --M 192 --K0 96 --n 60 > $T/pipeline_j1_192.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_timing.py --n 48 --beams 288/144 > $T/timing2.log 2>&1
echo done > $T/done2
