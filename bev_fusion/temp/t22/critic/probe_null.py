"""Null baseline of the distillation numbers: the per-cell mean of the teacher outputs over train frames, scored on val frames (same terms, same bands as validate())."""
import os, sys, types
import numpy as np
import torch
torch.set_num_threads(2)
sys.path.insert(0, "training")
import distill_student as D
from feature_cache import FeatureStore
import splits
s16, s8 = FeatureStore("data/cache/cam_feats"), FeatureStore("data/cache/cam_feats_bevfusion")
sp = splits.load("small")
tr = D.split_tokens(s16, sp["train"], "data/cache/img_front")[::4]
va = D.split_tokens(s16, sp["val"], "data/cache/img_front")[::2]
print(len(tr), len(va))
def load(toks):
    return {k: torch.stack([ (s16 if k.endswith("16") else s8).get(t)[0 if k[0] == "p" else 1] for t in toks]).float() for k in ("p16", "c16", "p8", "c8")}
A, B = load(tr), load(va)
null = {k: A[k].mean(0, keepdim=True).expand_as(B[k]) for k in A}
out = {"logit16": null["p16"].clamp_min(1e-8).log(), "logit8": null["p8"].clamp_min(1e-8).log(), "ctx16": null["c16"], "ctx8": null["c8"]}
maps = D.cell_maps(out, B)
mock = types.SimpleNamespace(grid8=torch.ones(1))
ys = D.head_rows(mock)
for name, m in maps.items():
    per_row = m.mean((0, 2)).numpy()
    ys_k = ys["16" if name.endswith("16") else "8"]
    bands = {b: float(per_row[idx].mean()) for b, idx in D.band_rows(ys_k).items()}
    print(name, "all", round(float(per_row.mean()), 3), {b: round(v, 3) for b, v in bands.items()})
print("head_rows 8", np.round(ys["8"], 1)[:12], "... n rows per band H8:", {b: len(i) for b, i in D.band_rows(ys["8"]).items()}, "H16:", {b: len(i) for b, i in D.band_rows(ys["16"]).items()})
