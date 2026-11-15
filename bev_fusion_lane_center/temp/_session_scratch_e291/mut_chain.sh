#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
python3 temp/e291/mutate.py cheap s02_prior_built_first s03_prior_flag_ignored s05_sigma_assert_off s09_data_arms_not_collated s10_partner_not_other_frame > temp/_session_scratch_e291/mut_phase1.log 2>&1
python3 temp/e291/mutate.py s01_term_not_added s04_trainer_without_prior_params s06_clean_training_spec s07_readouts_share_table s08_dump_overwrites > temp/_session_scratch_e291/mut_phase2.log 2>&1
echo done >> temp/_session_scratch_e291/mut_phase2.log
