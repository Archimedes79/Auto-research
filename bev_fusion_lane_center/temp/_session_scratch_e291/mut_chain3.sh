#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
python3 temp/e291/mutate.py cheap > temp/_session_scratch_e291/mut_p.log 2>&1
echo "rc $?" >> temp/_session_scratch_e291/mut_p.log
python3 temp/e291/mutate.py s04_trainer_without_prior_params s05_sigma_assert_off s06_clean_training_spec s07_readouts_share_table s08_dump_overwrites > temp/_session_scratch_e291/mut_s.log 2>&1
echo "rc $?" >> temp/_session_scratch_e291/mut_s.log
