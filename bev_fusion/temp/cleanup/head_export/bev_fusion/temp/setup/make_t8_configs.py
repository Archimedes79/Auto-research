"""One-off: the two arms of the feature swap (plan 2.8 step 0): t2_ref_s0 / t2_ref_s1 with the camera features from the BEVFusion camera-only cache.
Everything else (3 frames, window memory, 3000 steps, seeds 0 / 1, scene lists) is the reference's, so the result is comparable with ref_s0 / ref_s1.
Run from the bev_fusion folder: python temp/setup/make_t8_configs.py"""
import json

for seed in (0, 1):
    c = json.load(open(f"training/configs/t2_ref_s{seed}.json"))
    c["name"] = f"t8_bevfeat_s{seed}"
    c["feature_cache"] = "data/cache/cam_feats_bevfusion"
    c["model"] = {**c["model"], "camera_features": "bevfusion"}
    json.dump(c, open(f"training/configs/t8_bevfeat_s{seed}.json", "w"), indent=1)
    print("wrote", f"training/configs/t8_bevfeat_s{seed}.json")
