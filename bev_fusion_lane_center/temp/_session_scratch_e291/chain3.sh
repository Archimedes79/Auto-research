#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
L=temp/_session_scratch_e291
while pgrep -f 'final_chain2' > /dev/null; do sleep 3; done
bash temp/e291/run_staged.sh test_e291_staged.py --only dump > $L/r3_dump.log 2>&1; echo "exit $?" >> $L/r3_dump.log
python3 temp/e291/mutate_e291.py p19_sc_counts_not_modulo p20_sc_partner_none_not_unknown p21_sc_geometry_from_partner g01_start_stage_conditional g02_sha_check_off g03_runs_check_off g04_resume_ignored g05_wrapper_takes_any_stage > $L/r3_mut_a.log 2>&1; echo "rc $?" >> $L/r3_mut_a.log
python3 temp/e291/mutate_e291.py s08_dump_no_name_check s11_dump_folders_before_the_pass s12_dump_no_max_suffix s13_partner_not_cached > $L/r3_mut_b.log 2>&1; echo "rc $?" >> $L/r3_mut_b.log
