"""Loop 17 / E138b: does the map-conditioned builder use the map at all? VAL frames (the 250-frame subset) with the same synthetic loss of perception as in its training (a fixed draw per frame), with an exact SD map (pose error 0 / 2 m), no map, a wrong map;
the champion (42 channels) on the same degraded frames as reference. Also the clean frames. Usage (gcar, GPU): p17_mapdiag.py"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p17_eval as EV  # noqa: E402
import p17_mapnet as MN  # noqa: E402

net0, cfg = EV.load_net("D7", "e123_D7_r3_f1", ["M=288", "K0=144"])
cfg.temporal = False
net0.cfg.temporal = False
netm, _ = EV.load_net("D7", "e123_D7_r3_f1", ["M=288", "K0=144"])
netm.cfg.temporal = False
MN.widen(netm)
netm.load_state_dict(torch.load(f"{C.OUT}/e123/e17_map_a/model.pt", map_location="cuda"), strict=False)
netm.eval()
ks = C.val_subset()
P = dict(np.load(f"{PD.D}/poses_val.npz"))
asm = EV.CHAMP_ASM
rows = []
for name, net, mode, sigma, deg, drop in (
    ("champion, clean frames", net0, "none", 0.0, False, True),
    ("champion, degraded frames", net0, "none", 0.0, True, True),
    ("map builder, clean, exact map", netm, "exact", 0.0, False, False),
    ("map builder, clean, no map", netm, "none", 0.0, False, False),
    ("map builder, degraded, exact map", netm, "exact", 0.0, True, False),
    ("map builder, degraded, map error 2 m", netm, "exact", 2.0, True, False),
    ("map builder, degraded, wrong map", netm, "wrong", 0.0, True, False),
    ("map builder, degraded, no map", netm, "none", 0.0, True, False),
):
    base = PD.BuilderData("val_clean", ks=ks, train=False, with_prev=False)
    ds = MN.MapWrap(base, "val", P, train=False, mode=mode, sigma=sigma, degrade_eval=deg, drop_map=drop)
    raw = []
    _, gts, _ = PI.predict(net, ds, list(range(len(ds))), taus=(0.5,), bs=4, workers=2, raw=raw)
    graphs = [PI.assemble(r, None, 0.7, cfg, **asm) for r in raw]
    c = C.card(graphs, gts)
    rows.append(dict(name=name, **{k: float(c[k]) for k in ("S", "F1@1.0", "F1@0.5", "J", "vert100", "false_len_frac", "lanes_found")}))
    print(f"{name:42s} S {c['S']:.3f} F1@1 {c['F1@1.0']:.3f} J {c['J']:.3f} found {c['lanes_found']:.2f} FL {c['false_len_frac']:.3f} v100 {c['vert100']:.1f}", flush=True)
json.dump(rows, open("/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e17/e138b_mapdiag.json", "w"), indent=1)
