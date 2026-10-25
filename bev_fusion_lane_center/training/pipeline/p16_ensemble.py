"""Loop 16b: inference-time ensemble of builder checkpoints (no training): the node arrays of two or three models are concatenated (parent / link indices shifted) and go through the same path-union assembly; the junction heat maps are averaged.
Usage (gcar, GPU): p16_ensemble.py --models e123_D7_r3_f1,e123_D7_r3_s2 [--taus 0.7,0.8]"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_timing as TM  # noqa: E402


def merge(raws, junc_avg=True):
    out = {}
    offs = np.cumsum([0] + [len(r["pos"]) for r in raws])
    for k in ("pos", "th_travel", "flag", "pi", "nu", "valid", "level", "pos_rev"):
        if k in raws[0]:
            out[k] = np.concatenate([r[k] for r in raws], 0)
    out["parent"] = np.concatenate([np.where(r["parent"] >= 0, r["parent"] + o, -1) for r, o in zip(raws, offs)]).astype(np.int64)
    if "junc" in raws[0]:
        out["junc"] = np.mean([r["junc"].astype(np.float32) for r in raws], 0).astype(np.float16) if junc_avg else raws[0]["junc"]
    lk = []
    for r, o in zip(raws, offs):
        lk += [(i + o, j + o, p) for (i, j, p) in r.get("link_p", [])]
    out["link_p"] = lk
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--taus", default="0.7")
    ap.add_argument("--M", type=int, default=288)
    ap.add_argument("--K0", type=int, default=144)
    a = ap.parse_args()
    names = a.models.split(",")
    ks = C.val_subset()
    raws, cfg0, gts = [], None, None
    for nm in names:
        net, cfg = TM.load_builder(a.M, a.K0, nm)
        ds = PD.BuilderData("val_clean", ks=ks, train=False, with_prev=cfg.temporal)
        raw = []
        _, g, _ = PI.predict(net, ds, ks, taus=(0.7,), bs=8, workers=2, raw=raw)
        raws.append(raw)
        cfg0, gts = cfg0 or cfg, g
        del net
        torch.cuda.empty_cache()
    res = {}
    for tau in [float(x) for x in a.taus.split(",")]:
        for label, sel in [(n, [i]) for i, n in enumerate(names)] + [("+".join(names), list(range(len(names))))]:
            gs = [PI.assemble(merge([raws[i][f] for i in sel]) if len(sel) > 1 else raws[sel[0]][f], None, tau, cfg0, **TM.ASM) for f in range(len(ks))]
            m = C.card(gs, gts)
            res[f"{label} @tau {tau}"] = {k: float(m[k]) for k in ("S", "F1@1.0", "F1@0.5", "J", "vert100", "false_len_frac", "lanes_found") if k in m}
            print(f"{label:60s} tau {tau}: " + " ".join(f"{k} {res[f'{label} @tau {tau}'][k]:.3f}" for k in ("S", "F1@1.0", "J", "vert100", "false_len_frac", "lanes_found")), flush=True)
    json.dump(res, open(f"{C.LADDER}/diag/ensemble_{'_'.join(n.split('_')[-1] for n in names)}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
