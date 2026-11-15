"""scratch: correct pose minus control (paired scene bootstrap) from tu_arrays.npz: recall@0.5 / precision@0.5 / IoU@0.5 and the label-point recall of the union rules against their wrong-pose controls"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lane_readout as LR  # noqa: E402

z = np.load(os.path.join(HERE, "tu_arrays.npz"))
cnt, names, sc, found, cls = z["cnt"], list(z["names"]), z["sc"], z["found"], z["cls"]
S = sc.max() + 1
bt = LR.Boot(S)
out = []
for rule in "abc":
    for n in (1, 4):
        v, c1, c2 = names.index(f"N{n}{rule}"), names.index(f"N{n}{rule}-shift3m"), names.index(f"N{n}{rule}-wrongmotion")
        for cn, c in (("shift 3 m", c1), ("wrong motion", c2)):
            b = lambda i: bt.resample(bt.scenes(cnt[:, i, 0], sc))   # noqa: E731
            A, B = b(v), b(c)
            for nm, fn in (("IoU@0.5", lambda x: LR.iou(x[:, 1])), ("precision@0.5", lambda x: LR.prec(x[:, 1])), ("recall@0.5", lambda x: LR.rec(x[:, 1]))):
                d = fn(A) - fn(B)
                pt = fn(cnt[:, v, 0].sum(0)[None])[0] - fn(cnt[:, c, 0].sum(0)[None])[0]
                lo, hi = LR.ci(d)
                out.append(f"N{n}{rule} minus its {cn:12s} control: {nm:14s} {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        # label points, all / occluded / far lateral
        k = cls[:, 0].astype(int)
        occ, ay = cls[:, 3] > 0, cls[:, 2]
        for cn2, m in (("all points", np.ones(len(k), bool)), ("occluded", occ), ("|y| 12-30 m", ay >= 12)):
            for cn, c in (("shift 3 m", c1), ("wrong motion", c2)):
                psc = sc[k]
                den = np.bincount(psc, m.astype(float), S)
                nv, nc = np.bincount(psc, found[v] * m, S), np.bincount(psc, found[c] * m, S)
                with np.errstate(divide="ignore", invalid="ignore"):
                    d = (bt.W @ nv) / (bt.W @ den) - (bt.W @ nc) / (bt.W @ den)
                lo, hi = LR.ci(d)
                out.append(f"N{n}{rule} minus its {cn:12s} control: label-point recall {cn2:12s} {nv.sum() / den.sum() - nc.sum() / den.sum():+.4f} [{lo:+.4f}, {hi:+.4f}]")
txt = "\n".join(out) + "\n"
open(os.path.join(HERE, "tu_extra_result.txt"), "w").write(txt)
print(txt)
