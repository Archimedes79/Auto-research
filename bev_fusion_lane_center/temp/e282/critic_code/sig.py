import os, sys
sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], ".."))
from autogt import lane_gt
s = lane_gt.signature()
print("signature", s, "version", lane_gt.version_of(s))
print(os.listdir(lane_gt.DATA)[:10])
