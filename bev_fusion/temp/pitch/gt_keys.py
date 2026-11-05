import sys
sys.path.insert(0, ".")
sys.argv = ["x"]
import evaluation.bev_gtcheck as G
samples, rows = G.table(24, "15efbde2d1", 0)
o = rows["oracle"]
for k in sorted(o):
    if k.startswith("det/dt+0.00/vehicle") or k.startswith("curve/dt+0.00/velocity") or k.startswith("curve/dt+0.00/lane_divider"):
        print(k, round(o[k], 4))
