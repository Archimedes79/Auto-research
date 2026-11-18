"""E379 step-0 measurement (scratch, CPU): is the token set of B' on TRAIN cleaner than on VAL (B' was trained on the TRAIN frames: a second stage trained on its TRAIN tokens meets a shift at VAL)? From the predicted-node caches of B' seed S
(runs/e365_cache/be292_a_<S>_K512_..., TRAIN every-10th frame = 841 frames, VAL all 873): per split over the valid tokens: AP of sigmoid(e) for 'assigned to a GT node' (threshold-free), the mean existence probability of the assigned and of the unassigned tokens,
the share of assigned tokens below p 0.5, the tokens per frame above p 0.5, GT-node coverage (n_gt_assigned / n_gt) and the mean 'gd' (distance of an assigned token to its GT node). usage: python gap_tokens.py [seed]"""
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score

R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import lane_ptok_data as LP  # noqa: E402
import lane_tokcache as LC  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
for split, step in (("train", 10), ("val", 1)):
    c = LC.TokCache(LP.cache_dir(split, seed, "e292_a"))
    n = len(c.tokens)
    idx = np.arange(0, n, step)
    s, y, e_all, a_p, u_p, gd = [], [], [], [], [], []
    n_gt = n_asg = 0
    above = []
    for i in idx:
        v = np.array(c.f["valid"][i])
        e = np.array(c.f["e"][i], float)[v]
        g = np.array(c.f["gt_id"][i])[v] >= 0
        p = 1 / (1 + np.exp(-e))
        s.append(p)
        y.append(g)
        a_p.append(p[g])
        u_p.append(p[~g])
        above.append(int((p >= 0.5).sum()))
        n_gt += int(c.f["n_gt"][i])
        n_asg += int(c.f["n_gt_assigned"][i])
        d = np.array(c.f["gd"][i], float)[v][g]
        gd.append(d[np.isfinite(d)])
    s, y = np.concatenate(s), np.concatenate(y)
    a_p, u_p, gd = np.concatenate(a_p), np.concatenate(u_p), np.concatenate(gd)
    print(f"{split:5s} seed {seed}: {len(idx)} frames, {len(y)} valid tokens, assigned share {y.mean():.3f} | AP of p(exist) {average_precision_score(y, s):.4f} | mean p: assigned {a_p.mean():.3f}, unassigned {u_p.mean():.3f} | "
          f"assigned with p < 0.5: {(a_p < 0.5).mean():.3f} | unassigned with p >= 0.5: {(u_p >= 0.5).mean():.3f} | tokens p >= 0.5 per frame {np.mean(above):.1f} | GT node coverage {n_asg / n_gt:.3f} | mean gd {gd.mean():.3f} m")
