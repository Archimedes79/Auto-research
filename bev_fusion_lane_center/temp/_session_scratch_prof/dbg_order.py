import sys, torch
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/training/grid"); sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/training/data")
import test_lane_ptok_gpu as T
for w in (0, 2):
    ref, got = T._orders(w, 0)
    print("workers", w, "equal", ref == got)
    for p in range(3):
        print(" pass", p, "ref", ref[p][0], "got", got[p][0], "same", ref[p] == got[p])
