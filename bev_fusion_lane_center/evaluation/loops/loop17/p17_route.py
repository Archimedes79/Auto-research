"""Loop 17 / E135: map assignment of the loop-16 chain on a failure series: builder (series input) -> assembly -> route input (points every <= 4 m) -> route head along the edges -> route success against the route points,
per phase (cl / fail), for SD maps without error and with the scene-constant pose error (sigma 2 m, protocol of E79 / E114 / E118). Same code path as p16_pipeline_eval.py (plain builder instead of CUDA graphs).
Writes runs/e17/<out>/route_<tag>_<route>.json.
Usage (gcar, GPU): p17_route.py --out e135_route --series dark_single,dark_maxdecay_0.95 --routes p16_f2hr_route_s0,p16_f2hr_route_s1 --sigmas 0,2 [--ckpt <builder checkpoint dir>] [--prev 1]"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p10_net as PN  # noqa: E402
import p10_route_eval2 as RE  # noqa: E402
import sd_lib as RT11  # noqa: E402  (scene_pose_error, sd_raster: moved from the loop-11 script)
import p16_common as C  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_route_diag as RD  # noqa: E402
import p16_time as PTM  # noqa: E402
import p16_pipeline as PP  # noqa: E402
import p17_eval as EV  # noqa: E402
import p8_lib as L  # noqa: E402
import sd_lib as SL  # noqa: E402

E17 = EV.E17


@torch.no_grad()
def run_series(net, cfg, ds, rnets, sigmas, tau, asm, step=4.0):
    P = ds.P17
    sdl, rptl = L.cache("sd", "val"), L.cache("rpts", "val")
    out = {(si, ri): [] for si in range(len(sigmas)) for ri in range(len(rnets))}
    meta = []
    graphs = []
    for i in range(len(ds)):
        b = PD_collate(ds[i])
        row = int(ds.rows[int(b["k"][0])])
        prior = PTM.prior_from_prev(net, b, train=False) if cfg.temporal else None
        with torch.autocast("cuda", dtype=torch.float16):
            res = net(b["x"].float(), gt=None, share=0.0, prior=prior)
        r = PI.to_numpy(res, 0)
        g, ax = PI.assemble(r, None, tau, cfg, aux=True, **asm)
        graphs.append(g)
        meta.append((row, int(ds.meta["phase"][i])))
        rp, sd = rptl[row], sdl[row]
        ok = len(ax["V"]) and rp is not None and len(rp) and sd is not None
        it = PP.route_input(r, ax, g, res["h"][0], step, r["pi"]) if ok else None
        for si, sg in enumerate(sigmas):
            ras = None
            if ok:
                if sg == 0:
                    sdm, _, _ = L.draw_sd(sd, row, 0.0, 0.0, 1)
                    ras = torch.from_numpy(SL.raster_sd(sdm))[None].cuda()
                else:
                    ras = torch.from_numpy(RT11.sd_raster(sd, row, int(P["scene"][row]), sg, P, False)).float()[None].cuda()
            for ri, rn in enumerate(rnets):
                if not ok:
                    out[(si, ri)].append(None)
                    continue
                n = len(it["xy"])
                tok = torch.from_numpy(it["tok"].astype(np.float32))[None].cuda()
                xy_t = torch.from_numpy(it["xy"])[None].cuda()
                cp = torch.from_numpy(it["cp"])[None].cuda()
                mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    p = torch.sigmoid(rn(tok, xy_t, cp, ras, mask).float())[0].cpu().numpy()
                edges = RE.edges_of(it)
                xy = it["xy"].astype(np.float64)
                rt = [(x_, y_, ln) for (x_, y_, ln) in edges if 0.5 * (p[x_] + p[y_]) >= 0.5]
                rs = L.route_success(RD.pts_of(xy, rt), rp)
                out[(si, ri)].append(None if rs is None else float(rs["succ"]))
    return out, meta, graphs


def PD_collate(item):
    import p16_data as PD

    b = PD.collate([item])
    return {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="D7")
    ap.add_argument("--name", default="e123_D7_r3_f1")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--set", nargs="*", default=["M=288", "K0=144"])
    ap.add_argument("--asm", nargs="*", default=[f"{k}={v}" for k, v in EV.CHAMP_ASM.items()])
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--series", required=True)
    ap.add_argument("--routes", default="p16_f2hr_route_s0,p16_f2hr_route_s1")
    ap.add_argument("--sigmas", default="0,2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prev", type=int, default=1)
    ap.add_argument("--mapnet", type=int, default=0, help="1: --ckpt is a map-conditioned builder (p17_mapnet.py)")
    ap.add_argument("--map_mode", default="exact")
    ap.add_argument("--map_sigma", type=float, default=0.0)
    ap.add_argument("--tag_suffix", default="", help="appended to the result file name")
    a = ap.parse_args()
    asm = {k: float(v) for k, v in (x.split("=") for x in a.asm)}
    net, cfg = EV.load_net(a.design, a.name, a.set)
    if a.mapnet:
        import p17_mapnet as MN

        MN.widen(net)
        a.prev = 0
    if a.ckpt:
        net.load_state_dict(torch.load(a.ckpt if a.ckpt.endswith(".pt") else f"{C.OUT}/e123/{a.ckpt}/model.pt", map_location="cuda"), strict=False)
    if not a.prev:
        cfg.temporal = False
        net.cfg.temporal = False
    sigmas = [float(s) for s in a.sigmas.split(",")]
    rnames = a.routes.split(",")
    rnets = []
    for rn in rnames:
        m = PN.RouteXNet().cuda()
        m.load_state_dict(torch.load(f"{L.OUT}/{rn}/model.pt", map_location="cuda"))
        rnets.append(m.eval())
    os.makedirs(f"{E17}/{a.out}", exist_ok=True)
    t0 = time.time()
    for tag in a.series.split(","):
        ds = EV.SeriesData(tag, with_prev=bool(cfg.temporal))
        if a.mapnet:
            ds = MN.MapWrap(ds, "val", ds.P17, train=False, mode=a.map_mode, sigma=a.map_sigma)
        res, meta, graphs = run_series(net, cfg, ds, rnets, sigmas, a.tau, asm)
        ph = np.asarray([m[1] for m in meta])
        card = dict(tag=tag, name=a.name, ckpt=a.ckpt, sigmas=sigmas, routes=rnames, prev=bool(cfg.temporal))
        for (si, ri), v in res.items():
            v = np.asarray([np.nan if x is None else x for x in v], float)
            for nm, mk in (("cl", ph == 0), ("fail", ph == 1)):
                x = v[mk]
                card[f"sigma{sigmas[si]:g}_{rnames[ri]}_{nm}"] = dict(route_success=float(np.nanmean(x)), n=int(np.isfinite(x).sum()), per_frame=[None if not np.isfinite(t) else t for t in v.tolist()] if nm == "fail" else None)
        json.dump(card, open(f"{E17}/{a.out}/route_{tag}{a.tag_suffix}.json", "w"))
        line = " | ".join(f"s{sigmas[si]:g} {rnames[ri][-2:]}: cl {card[f'sigma{sigmas[si]:g}_{rnames[ri]}_cl']['route_success']:.3f} fail {card[f'sigma{sigmas[si]:g}_{rnames[ri]}_fail']['route_success']:.3f}" for (si, ri) in res)
        print(f"[{tag}] {line}  {time.time() - t0:.0f} s", flush=True)
    print("DONE", time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
