#!/bin/bash
# scratch: rebuild the big report, sync the result folder, make the proofreading text, run the numbers check, build the mail body
cd /mnt/c/Auto-research || exit 1
bash bev_fusion/temp/t16/sync_result_folder.sh 2>&1 | tail -n 3
python3 bev_fusion/temp/t16/html2txt.py bev_fusion/results/t16/report_big.html bev_fusion/results/t16/report_big.txt
bash bev_fusion/temp/t16/check_report_numbers.sh
python3 bev_fusion/temp/t16/build_mail2.py > bev_fusion/results/BEV_result_report/mail.txt && wc -c bev_fusion/results/BEV_result_report/mail.txt
