"""The complete fast path (CUDA-graph builder runtime + fast assembly) on ALL validation frames against the numbers of the eager final evaluation. Usage: p16_fast_val.py --name e123_D7_r3_f1 --M 288 --K0 144"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_runtime as RT  # noqa: E402
import p16_timing as TM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="e123_D7_r3_f1")
ap.add_argument("--M", type=int, default=288)
ap.add_argument("--K0", type=int, default=144)
ap.add_argument("--eager1", action="store_true", help="eager forward at batch 1 instead of the CUDA graphs (same assembly)")
a = ap.parse_args()
net, cfg = TM.load_builder(a.M, a.K0, a.name)
ds = PD.BuilderData("val_clean", ks=None, train=False, with_prev=cfg.temporal)
n = len(ds)
b0 = TM.to_gpu(PD.collate([ds[0]]))
fb = RT.FastBuilder(net, cfg, b0)
graphs, gts, t_all = [], [], []
t0 = time.time()
for i in range(n):
    b = TM.to_gpu(PD.collate([ds[i]]))
    t1 = time.time()
    if a.eager1:
        import p16_time as PTM

        pr = PTM.prior_from_prev(net, b, train=False) if cfg.temporal else None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            res = net(b["x"].float(), gt=None, share=0.0, prior=pr)
        r = PI.to_numpy(res, 0)
    else:
        r = fb.raw(b)
    g = PI.assemble(r, None, 0.7, cfg, **TM.ASM)
    t_all.append(time.time() - t1)
    graphs.append(g)
    gts.append(ds.gt[ds.rows[int(b["k"][0])]])
m = C.card(graphs, gts)
print(f"{a.name} M{a.M}/{a.K0}: frames {n}, graph production {np.mean(t_all) * 1e3:.1f} ms mean, {np.median(t_all) * 1e3:.1f} ms median (data loading excluded), total loop {time.time() - t0:.0f} s")
print("fast path on all validation frames:", {k: round(float(m[k]), 4) for k in ("S", "F1@1.0", "F1@0.5", "J", "vert100", "false_len_frac", "lanes_found") if k in m})
ref = json.load(open(f"{C.LADDER}/cards/{a.name}.finalA_val_clean.json", encoding="utf-8-sig")) if os.path.exists(f"{C.LADDER}/cards/{a.name}.finalA_val_clean.json") else None
if ref and ref.get("sweep", {}).get("0.7"):
    print("eager final evaluation (card):", {k: round(float(v), 4) for k, v in ref["sweep"]["0.7"].items() if k in ("S", "F1@1.0", "F1@0.5", "J", "vert100", "false_len_frac", "lanes_found")})
json.dump({k: float(m[k]) for k in ("S", "F1@1.0", "F1@0.5", "J", "vert100", "false_len_frac", "lanes_found") if k in m}, open(f"{C.LADDER}/diag/fastval_{a.name}_{a.M}{'_eager1' if a.eager1 else ''}.json", "w"), indent=1)
