"""Hysteresis threshold in the assembly (no training): nodes between tau_low and tau extend admitted nodes. Usage: p16_hyst.py --name e123_D7_r3_f1"""
import argparse
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_timing as TM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="e123_D7_r3_f1")
a = ap.parse_args()
net, cfg = TM.load_builder(288, 144, a.name)
ks = C.val_subset()
ds = PD.BuilderData("val_clean", ks=ks, train=False, with_prev=cfg.temporal)
raw = []
_, gts, _ = PI.predict(net, ds, ks, taus=(0.7,), bs=8, workers=2, raw=raw)
res = {}
for tau, tau_low in itertools.product((0.7, 0.8, 0.9, 0.95), (0.0, 0.5)):
    if tau_low and tau_low >= tau:
        continue
    gs = [PI.assemble(r, None, tau, cfg, tau_low=tau_low, **TM.ASM) for r in raw]
    m = C.card(gs, gts)
    key = f"tau {tau} tau_low {tau_low}"
    res[key] = {k: float(m[k]) for k in ("S", "F1@1.0", "F1@0.5", "J", "vert100", "false_len_frac", "lanes_found", "R@1.0") if k in m}
    print(f"{key:24s} " + " ".join(f"{k} {res[key][k]:.3f}" for k in ("S", "F1@1.0", "J", "vert100", "false_len_frac", "lanes_found")), flush=True)
json.dump(res, open(f"{C.LADDER}/diag/hyst_{a.name[-2:]}.json", "w"), indent=1)
