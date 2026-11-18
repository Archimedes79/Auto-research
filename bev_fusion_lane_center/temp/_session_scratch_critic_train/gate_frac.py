"""scratch (critic): fraction of the K x K pairs inside the allowed gate on real cached frames (CPU, 40 val frames, one thread)"""
import os, sys
import numpy as np
import torch
torch.set_num_threads(1)
R = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(R, "training", "grid")); sys.path.insert(0, os.path.join(R, "training")); sys.path.insert(0, os.path.join(R, "training", "data")); sys.path.insert(0, os.path.join(R, "training", "data", "prep")); sys.path.insert(0, os.path.join(R, "evaluation"))
import jtok as JT
d = os.path.join(R, "runs", "e365_cache", "be292_a_0_K512_9ad1d4ce30_v1", "val")
pos, dirv, e, valid = (np.load(os.path.join(d, n + ".npy"), mmap_mode="r") for n in ("pos", "dirv", "e", "valid"))
print("shapes", pos.shape, dirv.shape, e.shape, valid.shape)
n = 40
p, dv, ee, v = (torch.from_numpy(np.array(a[:n])) for a in (pos, dirv, e, valid))
g, allowed = JT.pair_geometry(p.float(), dv.float(), v.bool(), e=ee.float().sigmoid())
K = p.shape[1]
chord = g[..., 2] * 20.0
print("valid tokens per frame: mean %.1f min %d max %d (K %d)" % (v.sum(1).float().mean(), v.sum(1).min(), v.sum(1).max(), K))
print("allowed pairs / K^2: mean %.4f ; / (valid^2): mean %.4f" % ((allowed.sum((1, 2)).float() / K ** 2).mean(), (allowed.sum((1, 2)).float() / v.sum(1).float() ** 2).mean()))
print("short-class share of allowed %.3f, chord<=8 m" % (((chord <= 8) & allowed).sum().float() / allowed.sum()))
