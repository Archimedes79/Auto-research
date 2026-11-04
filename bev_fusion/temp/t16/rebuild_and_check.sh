#!/bin/bash
# scratch: rebuild the big report, sync the result folder, then the first-pass numbers check of the report against the stored sources (advisory, lists what it cannot find)
cd /mnt/c/Auto-research || exit 1
bash bev_fusion/temp/t16/sync_result_folder.sh 2>&1 | tail -n 3
python3 bev_fusion/temp/t16/html2txt.py bev_fusion/results/t16/report_big.html bev_fusion/results/t16/report_big.txt
T=bev_fusion/temp/t16
SRC="$T/history_digest.md $T/compare_t17.txt $T/compare_t18.txt $T/compare_t18_vs_base.txt $T/compare_ctl.txt $T/rule_t18.txt $T/rule_t17.txt $T/soa_digest.md $T/lane_reading.html bev_fusion/records/LEDGER.csv"
SRC="$SRC $(ls $T/soa/*.md $T/soa/*.json 2>/dev/null | tr '\n' ' ') $(ls bev_fusion/runs/t16_full_s0/eval/by_n/table_n1000.txt bev_fusion/runs/t17_data/eval/by_n/table_n1000.txt bev_fusion/runs/t18_fix/eval/by_n/table_n1000.txt bev_fusion/runs/*/eval/table_train.txt bev_fusion/runs/t1*/eval/timing_modules.json 2>/dev/null | tr '\n' ' ')"
SRC="$SRC $(ls bev_fusion/experiments/t1[6-8]*/experiment.md | tr '\n' ' ')"
python3 autoresearch/tools/check_numbers.py bev_fusion/results/t16/report_big.html $SRC --show 80 > bev_fusion/temp/t16/check_numbers_report.txt 2>&1
head -n 3 bev_fusion/temp/t16/check_numbers_report.txt
wc -l bev_fusion/temp/t16/check_numbers_report.txt
