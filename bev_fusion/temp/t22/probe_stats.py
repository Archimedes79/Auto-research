import os, sys, json, glob, time
sys.path.insert(0, "training")
import numpy as np, torch
torch.set_num_threads(2)
from feature_cache import FeatureStore
from lane_branch import load_image
t0 = time.time()
s16, s8 = FeatureStore("data/cache/cam_feats"), FeatureStore("data/cache/cam_feats_bevfusion")
print("stores", len(s16), len(s8), time.time() - t0)
sc = json.load(open("splits/small.json"))["train"][0]
toks = [str(t) for t in np.load("data/cache/cam_feats/%s.index.npz" % sc)["tokens"]]
toks = [t for t in toks if os.path.exists("data/cache/img_front/%s/%s.jpg" % (t[:2], t))][:6]
for tok in toks[:3]:
    t0 = time.time()
    p16, c16 = s16.get(tok); p8, c8 = s8.get(tok); im = load_image("data/cache/img_front", tok)
    dt = time.time() - t0
    p16, c16, p8, c8 = p16.float(), c16.float(), p8.float(), c8.float()
    ent = lambda p: float(-(p * torch.log(p.clamp_min(1e-8))).sum(0).mean())
    print(tok[:8], im.shape, im.dtype, "read s %.3f" % dt)
    print("  16: psum %.4f ent %.2f ctx mean %.3f ms %.3f std %.3f | 8: psum %.4f ent %.2f ctx mean %.3f ms %.3f std %.3f" % (
        float(p16.sum(0).mean()), ent(p16), float(c16.mean()), float(c16.pow(2).mean()), float(c16.std()),
        float(p8.sum(0).mean()), ent(p8), float(c8.mean()), float(c8.pow(2).mean()), float(c8.std())))
    print("  KL(t||uniform) 16: %.3f  8: %.3f (log118=%.3f)" % (float((p16 * (torch.log(p16.clamp_min(1e-8)) + np.log(118))).sum(0).mean()), float((p8 * (torch.log(p8.clamp_min(1e-8)) + np.log(118))).sum(0).mean()), np.log(118)))
