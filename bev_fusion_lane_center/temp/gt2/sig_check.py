import os, sys
R = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.dirname(R))
from GtGenerator import lane_gt
sig = lane_gt.signature()
d = os.path.join(lane_gt.DATA, sig)
print("lane signature now:", sig, "version", lane_gt.version_of(sig), "store", d, "exists", os.path.isdir(d), "frames", len(lane_gt.held(d)))
from GtGenerator import lane_gt as L
tok = sorted(lane_gt.held(d))[0]
g = lane_gt.load(tok)
print("loaded a frame:", len(g["pieces"]), "pieces", g.get("signature"))
