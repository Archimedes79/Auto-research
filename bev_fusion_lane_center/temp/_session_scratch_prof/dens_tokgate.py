"""scratch: how dense is the allowed pair gate on real frames (B' cache seed 0, 64 TRAIN frames), per class"""
import os, sys, numpy as np, torch
R = os.environ["BFLC_ROOT"]
for s in ("training", "training/grid", "training/data", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(R, s))
import jtok as JT, lane_ptok_data as LP
torch.set_num_threads(2)
ds, dl = LP.loader("train", 0, 8, 0)
fr, ch, vs, mx = [], [], [], []
for i in range(0, 64, 8):
    b = LP.collate([ds[j] for j in range(i, i + 8)])
    pos, dirv, valid, e = (b[k].cuda() for k in ("pos", "dirv", "valid", "e"))
    g, al = JT.pair_geometry(pos.float(), dirv.float(), valid.bool(), e=e.float().sigmoid())
    K = al.shape[-1]
    nv = valid.sum(1).float()
    fr.append((al.flatten(1).sum(1).float() / (K * K)).mean().item())
    vs.append(nv.mean().item())
    mx.append(al.sum(-1).float().max().item())
    ch.append((al.flatten(1).sum(1).float() / (nv * nv)).mean().item())
print(dict(K=K, valid_tokens_per_frame=round(float(np.mean(vs)), 1), allowed_share_of_K2=round(float(np.mean(fr)), 4), allowed_share_of_valid_pairs=round(float(np.mean(ch)), 4),
           allowed_pairs_per_frame=round(float(np.mean(fr)) * K * K), max_allowed_per_source=int(max(mx))))
