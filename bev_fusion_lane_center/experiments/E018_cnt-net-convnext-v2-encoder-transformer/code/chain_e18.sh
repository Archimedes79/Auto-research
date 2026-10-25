#!/usr/bin/env bash
# startet E18 (CNT-Net) erst, wenn das U-Netz-Training (B) sein Modell geschrieben hat (GPU-Speicher reicht nicht fuer zwei Trainings)
until [ -f /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/_loop1_crn/b123_unet64/heatnet.pt ]; do sleep 20; done
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/scripts/run_crn_e18.sh
