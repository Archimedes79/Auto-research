"""Loop 16: quality of the junction head ALONE: peaks of the split / merge heat maps against the ground-truth fork / merge landmarks (6 m tolerance, one-to-one), and what an oracle anchoring would give.
Usage (gcar, GPU): p16_junc_head_eval.py --design D7 --name e123_D7_r3_f1 [--set M=288 K0=144] -> prints P / R / F1 per threshold; out/loop2/e123_ladder/diag/<name>_junchead.json"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.ndimage import maximum_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_train as PT  # noqa: E402


def peaks(m, thr, size=9):
    pk = (m == maximum_filter(m, size=size)) & (m >= thr)
    ij = np.argwhere(pk)
    return np.array([[(i + 0.5) * 0.5 - 50.0, (j + 0.5) * 0.5 - 50.0] for i, j in ij], np.float64).reshape(-1, 2), (m[pk] if len(ij) else np.zeros(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=250)
    a = ap.parse_args()
    c = json.load(open(f"{C.LADDER}/cards/{a.name}.json", encoding="utf-8-sig"))
    extra = dict(c["cfg"])
    extra.update(PT.parse_set(a.set))
    net, cfg = PT.build(a.design, extra)
    assert cfg.junc, "the run has no junction head"
    net = net.cuda()
    net.load_state_dict(torch.load(f"{C.OUT}/e123/{a.name}/model.pt", map_location="cuda"), strict=False)
    ks = C.val_subset()[:: max(1, 250 // a.n)]
    ds = PD.BuilderData("val_clean", ks=ks, train=False, with_prev=cfg.temporal)
    raw = []
    _, gts, _ = PI.predict(net, ds, ks, taus=(0.5,), bs=4, workers=2, raw=raw)
    out = {}
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        res = {}
        for kind, nm in ((0, "split"), (1, "merge")):
            m_ = n_p = n_g = 0
            for r, g in zip(raw, gts):
                P, _ = peaks(r["junc"][kind].astype(np.float32), thr)
                G = GE.junctions(g)[kind]
                m_ += GE._match_points(P, G, 6.0)
                n_p += len(P)
                n_g += len(G)
            Pp, Rr = m_ / max(n_p, 1), m_ / max(n_g, 1)
            res[nm] = dict(P=Pp, R=Rr, F1=2 * Pp * Rr / max(Pp + Rr, 1e-9), n_pred=n_p / len(raw), n_gt=n_g / len(raw))
        out[str(thr)] = res
        print(f"thr {thr:.1f} | split P {res['split']['P']:.2f} R {res['split']['R']:.2f} F1 {res['split']['F1']:.2f} (pred {res['split']['n_pred']:.1f} / gt {res['split']['n_gt']:.1f}) | merge P {res['merge']['P']:.2f} R {res['merge']['R']:.2f} F1 {res['merge']['F1']:.2f}")
    json.dump(out, open(f"{C.LADDER}/diag/{a.name}_junchead.json", "w"), indent=1)


if __name__ == "__main__":
    main()
