import os, pickle, sys
R = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(R, "evaluation"))
import qgraph_eval as QE
ld = lambda n: pickle.load(open(os.path.join(R, "runs/e312_p_s0", n + ".counts.pkl"), "rb"))
for n in ("ends_card_snap_thr0.1", "ends_card_snap_nogeom_thr0.1", "ends_card_cnt_thr0.1", "ends_card_a_snap_thr0.1", "ends_card_c2_snap_thr0.1"):
    g = QE.graph_numbers(ld(n)["head"])
    print(n, "topo", round(g["topo_F1"], 4), "dup", round(g["dup_frac"], 4), "junc", round(g["junction_F1"], 4))
