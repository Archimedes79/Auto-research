import json, os, sys, glob
import numpy as np
import torch
torch.set_num_threads(2)
sys.path.insert(0, "training")
from feature_cache import FeatureStore
import splits
from lane_branch import image_path
s16, s8 = FeatureStore("data/cache/cam_feats"), FeatureStore("data/cache/cam_feats_bevfusion")
print("stores", len(s16), len(s8), s16.feature_hw(), s8.feature_hw())
for nm in ("small", "large"):
    sp = splits.load(nm)
    print(nm, len(sp["train"]), len(sp["val"]))
sm, lg = splits.load("small"), splits.load("large")
print("small.train & large.val", len(set(sm["train"]) & set(lg["val"])), " small.val & large.train", len(set(sm["val"]) & set(lg["train"])), "small.val & large.val", len(set(sm["val"]) & set(lg["val"])))
shard_scene = [os.path.basename(p)[:-4] for p in s16.shards]
for nm in ("small", "large"):
    sp = splits.load(nm)
    for part in ("train", "val"):
        want = set(sp[part])
        toks = [t for t, (sid, _) in s16.index.items() if shard_scene[sid] in want and os.path.exists(image_path("data/cache/img_front", t))]
        allt = [t for t, (sid, _) in s16.index.items() if shard_scene[sid] in want]
        print(nm, part, "key frames with jpeg", len(toks), "all tokens in shards", len(allt), "missing from s8", sum(1 for t in toks if not s8.has(t)))
# stats on one shard
sp = splits.load("small")
toks = [t for t, (sid, _) in s16.index.items() if shard_scene[sid] == sp["val"][0] and os.path.exists(image_path("data/cache/img_front", t))][:6]
for k, st in (("16", s16), ("8", s8)):
    P = torch.stack([st.get(t)[0] for t in toks]).float(); C = torch.stack([st.get(t)[1] for t in toks]).float()
    print("H"+k, P.shape, C.shape, "prob sum min/max", float(P.sum(1).min()), float(P.sum(1).max()), "ctx mean sq", float(C.pow(2).mean()), "ctx mean", float(C.mean()), "ctx absmax", float(C.abs().max()), "per-cell ctx norm mean", float(C.norm(dim=1).mean()))
    ent = -(P.clamp_min(1e-8) * P.clamp_min(1e-8).log()).sum(1)
    print("   entropy per row (mean over cols/samples):", [round(float(x), 2) for x in ent.mean((0, 2))])
    am = P.argmax(1).float()
    print("   mean argmax bin per row:", [round(float(x), 1) for x in am.mean((0, 2))])
    # KL of uniform/ mean-prob baseline
