import sys, os
for p in ("training/train", "training/grid", "training/data", "evaluation", "training"):
    sys.path.insert(0, os.path.join("/mnt/c/Auto-research/bev_fusion_lane_center", p))
import torch, qgates as G, qpredict as QP
class R:
    def log(self, *a): print("LOG", *a)
m = QP.load_model("/mnt/c/Auto-research/bev_fusion_lane_center/runs/e273_smoke2", "cuda")
m.train()
for k in (4, 5):
    r = G.gate(m, k, R(), n=8)
    print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items() if a != "flag_recall"})
