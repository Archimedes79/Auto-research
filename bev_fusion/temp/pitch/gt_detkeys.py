import sys
sys.path.insert(0, ".")
import evaluation.bev_gtcheck as G
samples, rows = G.table(12, "15efbde2d1", 0)
o = rows["oracle"]
for k in sorted(o):
    if k.startswith("det/dt+0.00/vehicle/all/") or k.startswith("det/dt+0.00/pedestrian/all/ap@2") or k.startswith("det/dt+0.00/mean"):
        print(k, round(o[k], 4))
