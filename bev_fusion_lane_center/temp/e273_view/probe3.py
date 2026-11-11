import os, json, sys, time, numpy as np
HOME = os.path.expanduser("~"); R = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(R, ".."))
from autogt import lane_gt, lane_graph_view as V
from autogt.light_nusc import light_nuscenes
toks = json.load(open(f"{HOME}/exp_gcar_b123/feats/val_tokens.json"))
sp = json.load(open(f"{R}/data/splits.json"))
root = f"{HOME}/data/nuscenes_full"
t0 = time.time()
for name in ("scene-0221", "scene-0108", "scene-0092", "scene-0013"):
    nusc = light_nuscenes(root, "v1.0-trainval", [name], cameras=V.CAMERAS, radars=False)
    print(name, "light nusc", round(time.time() - t0, 1))
    class GT: pass
    gt = GT(); gt.nusc = nusc; gt.dataroot = root
    sc = nusc.scene[0]
    tk = sc["first_sample_token"]
    ok = V.has_cameras(gt, tk)
    print(name, "cameras on disk (first keyframe):", ok)
