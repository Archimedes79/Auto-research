sed -i 's|c.set_title("(c) per scene: pooled topo F1 (11, 1, 15, 7, 42 are the weakest)", fontsize=9)|c.set_title("(c) per scene: pooled topo F1 (weakest: " + ", ".join(str(s) for s in order[:5]) + ")", fontsize=9)|' temp/qual_report_1115/overview_q1115.py
grep -n "weakest" temp/qual_report_1115/overview_q1115.py
