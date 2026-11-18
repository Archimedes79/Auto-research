"""E369 probe (throw-away): c-rule links against the head's candidate gate on VAL cache frames; kept counts; rule time per batch; e near 0"""
import os
import sys
import time

import numpy as np
import torch

ROOT = os.environ["BFLC_ROOT"]
for s in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train"):
    sys.path.insert(0, os.path.join(ROOT, s))
t0 = time.time()
import jtok_eval as JV  # noqa: E402
print(f"import jtok_eval {time.time() - t0:.2f} s")
import jtok as JT  # noqa: E402
import jtoken_eval as E  # noqa: E402

torch.set_num_threads(2)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
split = sys.argv[2] if len(sys.argv) > 2 else "val"
ds = E.open_cache(split, 0)
idx = E.sample(len(ds), split, 3)[:n] if split == "train" else np.arange(n)
tot = dict(links=0, out_gate=0, cls_flip=0, long=0, kept=0, near0=0, keepdiff=0, frames=0)
ms = []
lstar = []
for b in E.batches(ds, idx, bs=8):
    t1 = time.time()
    for i, t in enumerate(b["tokens_np"]):
        keep, e, lk = E.rule_frame(t)
    ms.append((time.time() - t1) * 1000)
    pos, dirv, valid = b["pos"].float(), b["dirv"].float(), b["valid"].bool()
    g, allowed = JT.pair_geometry(pos, dirv, valid, e=b["e"].float().sigmoid())
    chord = g[..., 2] * 20.0
    for i, t in enumerate(b["tokens_np"]):
        keep, e, lk = E.rule_frame(t)
        s, d = np.asarray(lk["src"]), np.asarray(lk["dst"])
        al = allowed[i].numpy()[s, d]
        lg = chord[i].numpy()[s, d] > JT.SHORT
        tot["links"] += len(s)
        tot["out_gate"] += int((~al).sum())
        tot["cls_flip"] += int((lg != lk["long"]).sum())
        tot["long"] += int(lk["long"].sum())
        tot["kept"] += int(keep.sum())
        ee = b["e"][i].numpy()
        tot["near0"] += int((np.abs(ee) < 1e-4).sum())
        k2 = t["valid"] & (torch.from_numpy(ee).sigmoid().numpy() >= 0.5)
        tot["keepdiff"] += int((k2 != keep).sum())
        lstar += np.log(torch.from_numpy(ee[d]).sigmoid().numpy() + 1e-3).tolist()
        tot["frames"] += 1
        if (~al).any() and tot["frames"] < 30:
            for k in np.flatnonzero(~al)[:2]:
                a_, c_ = s[k], d[k]
                ch = t["pos"][c_] - t["pos"][a_]
                print("out of gate: L", np.linalg.norm(ch), "dot ta", ch @ t["dirv"][a_], "dot tb", ch @ t["dirv"][c_], "turn", t["dirv"][a_] @ t["dirv"][c_], "long", lk["long"][k])
print(tot)
print(f"out-of-gate share {tot['out_gate'] / tot['links']:.5f}, class flips {tot['cls_flip']}, kept per frame {tot['kept'] / tot['frames']:.1f}, rule ms per batch of 8: median {np.median(ms):.1f} max {np.max(ms):.1f}")
ls = np.array(lstar)
print("l* = log(w_dst + 1e-3): quantiles 5/25/50/75/95", np.percentile(ls, [5, 25, 50, 75, 95]).round(3))
