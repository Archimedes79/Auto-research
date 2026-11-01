"""throw-away probe: the builder of HEAD (before E266) and the edited builder give bit-identical raw outputs and graphs on real VAL frames of
`recall` (CPU, default config: lattice_ring 0)"""
import subprocess
import sys
import types

import numpy as np
import torch

import evaluate as EV
import p16_data as PD
import p16_infer as PI
import p16_train as PT

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
src = subprocess.run(["git", "-c", "safe.directory=*", "-C", EV.R, "show", "HEAD:bev_fusion_lane_center/training/models/p16_builder.py"], capture_output=True, text=True, check=True).stdout
old = types.ModuleType("p16_builder_head")
exec(compile(src, "p16_builder_head", "exec"), old.__dict__)
e = EV.registry()["recall"]
extra = EV.entry_cfg(e)
net_new, cfg = PT.build(e["design"], extra)
kw = dict(PT.DESIGNS[e["design"]])
kw.update(extra)
net_old = old.Builder(old.DefaultCfg(**{k: v for k, v in kw.items() if k != "w"}))
sd = torch.load(f"{EV.R}/runs/e123/{e['run']}/model.pt", map_location="cpu")
for n in (net_new, net_old):
    n.load_state_dict(sd, strict=False)
ds = PD.BuilderData("val_clean", ks=list(range(N)), train=False, with_prev=cfg.temporal)
out = {}
for lab, n in (("new", net_new), ("head", net_old)):
    torch.manual_seed(0)
    raw = []
    g, _, _ = PI.predict(n, ds, list(range(N)), taus=(0.7,), bs=2, workers=0, dev="cpu", raw=raw)
    out[lab] = (raw, g[0.7])
bad = []
for k, (a, b) in enumerate(zip(out["new"][0], out["head"][0])):
    for key in a:
        x, y = a[key], b[key]
        same = (x == y) if isinstance(x, list) else (x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y))
        if not same:
            bad.append((k, key))
ga, gb = out["new"][1], out["head"][1]
gsame = all(p["edges"] == q["edges"] and len(p["pts"]) == len(q["pts"]) and all(np.array_equal(u, v) for u, v in zip(p["pts"], q["pts"])) for p, q in zip(ga, gb))
print(f"{N} frames, fields {sorted(out['new'][0][0])}: raw differences {bad[:10]} ({len(bad)}), graphs identical {gsame}, nodes per frame {len(out['new'][0][0]['pi'])}")
