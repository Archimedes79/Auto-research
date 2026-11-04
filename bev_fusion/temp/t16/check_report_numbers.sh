#!/bin/bash
# scratch: the first-pass numbers check of the big report against the stored sources (advisory; lists the decimal numbers found in no source)
cd /mnt/c/Auto-research || exit 1
T=bev_fusion/temp/t16
SRC=""
for f in $T/history_digest.md $T/compare_t17.txt $T/compare_t18.txt $T/compare_t18_vs_base.txt $T/rule_t18.txt $T/rule_t17.txt $T/soa_digest.md $T/lane_reading.html bev_fusion/records/LEDGER.csv \
         $T/soa/*.md $T/soa/*.json bev_fusion/runs/t16_full_s0/eval/by_n/table_n1000.txt bev_fusion/runs/t17_data/eval/by_n/table_n1000.txt bev_fusion/runs/t18_fix/eval/by_n/table_n1000.txt \
         bev_fusion/runs/t1*/eval/table_train.txt bev_fusion/runs/t1*/eval/timing_modules.json bev_fusion/experiments/t1[6-8]*/experiment.md bev_fusion/temp/t16/report_text.py; do
  [ -f "$f" ] && SRC="$SRC $f"
done
python3 autoresearch/tools/check_numbers.py bev_fusion/results/t16/report_big.html $SRC --show 120 > $T/check_numbers_report.txt 2>&1
wc -l $T/check_numbers_report.txt
head -n 4 $T/check_numbers_report.txt | cut -c1-200
