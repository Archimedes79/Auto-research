#!/bin/bash
# timing window: runs when the GPU is idle (after the seed-1 route-head trainings), then releases the point-spacing trainings (timing_ok)
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/diag_resample
T=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/timing
mkdir -p $T
rm -f $T/done
while [ ! -f $O/done_s1 ]; do sleep 20; done
sleep 20
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_pareto.py --n 60 > $T/pareto.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_pipeline.py --name e123_D7_r3_f1 --n 60 > $T/pipeline_f1.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_pipeline.py --name e123_D7_r3_j1 --n 60 > $T/pipeline_j1.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_timing.py --n 48 --beams 288/144 > $T/timing.log 2>&1
echo done > $T/done
touch $O/timing_ok
