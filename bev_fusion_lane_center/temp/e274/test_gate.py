import os, sys, torch
R = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(R, sub))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import qgates as QG, qpredict as QP
m = QP.load_model(os.path.join(R, "runs", "e273_p_s0"), "cpu")
QG.GATE_CFG.update(node_prec_min=0.75, node_rec_min=0.90)
r = QG.gate_nodes(m, None, 16)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "flag_recall"})
