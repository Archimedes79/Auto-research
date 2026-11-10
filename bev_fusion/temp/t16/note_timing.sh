#!/bin/bash
# scratch: judgement columns of timing.idle3 (all three timings were refused by the idle guard) in the BEV ledger
cd /mnt/c/Auto-research || exit 1
export AR_PROJECT=../bev_fusion
python3 autoresearch/tools/ledger.py note timing.idle3 --result "no number: the idle guard refused all three timings (t16_full_s0, t17_data, t18_fix) because my own CPU-only paired comparison (bev_suite.py compare) ran at the same time; the job still ended with exit 0, so the status 'ok' means only that the script ran" --decision "diagnostic run, feeds no verdict; repeated as timing.idle3b after the comparison ended"
