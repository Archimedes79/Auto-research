p = "evaluation/qgraph_run_eval.py"
s = open(p, encoding="utf-8").read()
s = s.replace("""f"| Q | {gr.get('Q', float('nan')):.4f} | | {dd['Q']:.4f} |", "",""", """f"| Q | {gr.get('Q', float('nan')):.4f} | | {dd['Q']:.4f} |",
         f"| vertices per 100 m: node ids / Douglas-Peucker 0.3 m (budget 14.3 on the DP basis) | {gr.get('vert100', float('nan')):.1f} / {gr.get('vert100_dp', float('nan')):.1f} | | {dd['vert100']:.1f} / |",
         f"| pieces per GT piece, id switches per 100 m | {gr.get('pieces_per_gt_piece', float('nan')):.3f}, {gr.get('id_switches_per_100m', float('nan')):.3f} | | |", "",""")
open(p, "w", encoding="utf-8").write(s)
