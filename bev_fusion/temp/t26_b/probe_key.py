import os
import GtGenerator
from GtGenerator import lane_gt
print(lane_gt.signature(rig="bev_fusion"), lane_gt.DATA)
k = GtGenerator.gt_key("bev_fusion", "v1.0-trainval", "full")
print(k)
print(k.part("lane").path)
