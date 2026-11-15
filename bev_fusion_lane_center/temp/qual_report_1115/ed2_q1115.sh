sed -i 's|b.set_title("(b) frame by frame, E324 s against B (above the line = E324 s better); numbers = pictures 1 - 9", fontsize=8)|b.set_title("(b) E324 s against B, frame by frame (above the line: E324 s better)", fontsize=9)|' temp/qual_report_1115/overview_q1115.py
sed -i 's|within 0.02: {873 - n_up - n_dn} frames"|within 0.02: {873 - n_up - n_dn} frames\nlabels 1 - 9 = pictures 1 - 9"|' temp/qual_report_1115/overview_q1115.py
grep -n "set_title(\"(b)\|labels 1" temp/qual_report_1115/overview_q1115.py
