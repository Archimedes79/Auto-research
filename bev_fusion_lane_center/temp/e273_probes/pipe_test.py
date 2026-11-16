import sys, time
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/training/data"); sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/training/grid")
import torch, numpy as np
import lane_gt_data as LD
ds = LD.QData("train", map_mode="train", pixels=True)
print("train frames", len(ds), "missing", ds.missing)
t = time.time()
items = [ds[i] for i in range(0, 40, 4)]
print("per item s", (time.time() - t) / len(items))
b = LD.collate(items[:4])
print({k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in b.items()})
