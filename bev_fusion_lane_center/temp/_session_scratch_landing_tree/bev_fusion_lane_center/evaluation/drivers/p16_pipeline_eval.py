"""The deployed path end to end on ALL validation frames: CUDA-graph builder -> fast assembly -> route input (points every <= step m) -> padded route-head graphs -> route success against the route points (exact map) and the graph metrics.
Usage (gcar, GPU): p16_pipeline_eval.py --name e123_D7_r3_j1 --M 192 --K0 96 --route p16_f2hr_route_s0"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p8_lib as L  # noqa: E402
import p10_net as PN  # noqa: E402
import p10_route_eval2 as RE  # noqa: E402
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_pipeline as PP  # noqa: E402
import p16_route_diag as RD  # noqa: E402
import p16_runtime as RT  # noqa: E402
import p16_timing as TM  # noqa: E402
import sd_lib as SL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="e123_D7_r3_f1")
ap.add_argument("--M", type=int, default=288)
ap.add_argument("--K0", type=int, default=144)
ap.add_argument("--route", default="p16_f2hr_route_s0")
ap.add_argument("--step", type=float, default=4.0)
ap.add_argument("--fork_prune", type=float, default=0.0, help="loop 18: junction-head threshold below which forks are un-linked (0 = off)")
ap.add_argument("--tag", default="", help="appended to the result file name")
a = ap.parse_args()
net, cfg = TM.load_builder(a.M, a.K0, a.name)
ds = PD.BuilderData("val_clean", ks=None, train=False, with_prev=cfg.temporal)
n = len(ds)
fb = RT.FastBuilder(net, cfg, TM.to_gpu(PD.collate([ds[0]])))
rnet = PN.RouteXNet().cuda()
rnet.load_state_dict(torch.load(f"{L.OUT}/{a.route}/model.pt", map_location="cuda"))
fr = PP.FastRoute(rnet)
part = ds.part if hasattr(ds, "part") else "val"
sdl, rptl = L.cache("sd", part), L.cache("rpts", part)
graphs, gts, succ, cov, prec, tt = [], [], [], [], [], []
for i in range(n):
    b = TM.to_gpu(PD.collate([ds[i]]))
    row = int(ds.rows[int(b["k"][0])])
    t0 = time.time()
    r = fb.raw(b)
    g, ax = PI.assemble(r, None, 0.7, cfg, aux=True, fork_prune=a.fork_prune, **TM.ASM)
    graphs.append(g)
    gts.append(ds.gt[row])
    rp, sd = rptl[row], sdl[row]
    if not len(ax["V"]) or rp is None or len(rp) == 0 or sd is None:
        tt.append(time.time() - t0)
        continue
    it = PP.route_input(r, ax, g, fb.h_static, a.step, r["pi"])
    sdm, _, _ = L.draw_sd(sd, row, 0.0, 0.0, 1)
    ras = torch.from_numpy(SL.raster_sd(sdm))[None].cuda()
    nn_ = len(it["xy"])
    p = fr(torch.from_numpy(it["tok"].astype(np.float32))[None].cuda(), torch.from_numpy(it["xy"])[None].cuda(), torch.from_numpy(it["cp"])[None].cuda(), ras, nn_).cpu().numpy()
    tt.append(time.time() - t0)
    edges = RE.edges_of(it)
    xy = it["xy"].astype(np.float64)
    rt = [(x_, y_, ln) for (x_, y_, ln) in edges if 0.5 * (p[x_] + p[y_]) >= 0.5]
    res = L.route_success(RD.pts_of(xy, rt), rp)
    if res is not None:
        succ.append(res["succ"])
        cov.append(res["cov"])
        prec.append(res["prec"])
m = C.card(graphs, gts)
out = dict(name=a.name, M=a.M, K0=a.K0, route=a.route, frames=n, route_frames=len(succ), route_success=float(np.mean(succ)), route_cov=float(np.mean(cov)), route_prec=float(np.mean(prec)), ms_per_frame_median=float(np.median(tt)) * 1e3, ms_per_frame_mean=float(np.mean(tt)) * 1e3,
           **{k: float(m[k]) for k in ("S", "F1@1.0", "F1@0.5", "J", "vert100", "false_len_frac", "lanes_found") if k in m})
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}))
json.dump(out, open(f"{C.LADDER}/diag/pipeeval_{a.name[-2:]}_{a.M}{a.tag}.json", "w"), indent=1)
