#!/bin/bash
# scratch: judgement columns of timing.idle3d in the BEV ledger, then the report rebuild + result folder sync
cd /mnt/c/Auto-research || exit 1
export AR_PROJECT=../bev_fusion
python3 autoresearch/tools/ledger.py note timing.idle3d --result "idle-GPU timing, batch 1, dts -0.5/0/+0.5, cold memory, median of 30 frames, all three marked idle (utilisation 2 - 5 percent, no other process): t16_full_s0 96.4 ms, t17_data 98.7 ms, t18_fix 99.7 ms per frame (files runs/<run>/eval/timing_modules.json); training micro-batch of the baseline 136 ms forward+loss / 92 ms backward" --decision "diagnostic run, feeds no verdict; made with temp/t16/timing_patched2.py: the frozen guard of bev_timing.py counts the timing process itself as a GPU user and refused three times on an idle GPU (open finding 2026-11-03), the wrapper drops the own pid and waits 3 s before the second look, every other idle check kept"
bash bev_fusion/temp/t16/sync_result_folder.sh 2>&1 | tail -n 25
