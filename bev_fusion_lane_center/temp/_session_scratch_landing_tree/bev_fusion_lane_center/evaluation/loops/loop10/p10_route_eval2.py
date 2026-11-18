"""Loop 10, E102 / E103: navigation attribute on stored tokens. Route probabilities per vertex under every SD-map cell of the loop-8 / 9 protocol -> route edges (mean probability of the two vertices >= tau) -> route success etc.
Metrics as p8_assign.decode_eval: route success (coverage and precision of the route points at 1.5 m >= 0.8, E59 definition), lane P / R / F1 of the traced edges, route-flagged length with a wrong / missing map.
Usage (gcar, GPU): p10_route_eval2.py --tok tok_e94_b_test_clean --split test_clean --model e103c [--kind graph | v1] [--taus 0.3,0.4,0.5]"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import p8_lib as L  # noqa: E402
import p10_net as PN  # noqa: E402
import sd_lib as SL  # noqa: E402

HOME = os.path.expanduser("~")
OUT = L.OUT
CELLS = {"exact": (0.0, 0.0, "n", {}), "s1": (1.0, 0.0, "n", {}), "s2": (2.0, 0.0, "n", {}), "s3": (3.0, 0.0, "n", {}), "s1d30": (1.0, 0.3, "n", {}), "wrong": (0.0, 0.0, "w", {}), "none": (0.0, 0.0, "0", {}),
         "a25": (2.5, 0.05, "n", {}), "bias1": (1.0, 0.0, "n", dict(bias=1.0)), "bias2": (1.0, 0.0, "n", dict(bias=2.0))}


def edges_of(it):
    """(a, b, length) of the consecutive vertices of every piece"""
    out = []
    xy = it["xy"].astype(np.float64)
    for path in it["pieces"]:
        for a, b in zip(path[:-1], path[1:]):
            out.append((a, b, float(np.linalg.norm(xy[b] - xy[a]))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--split", default="test_clean")
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", default="graph", help="graph (RouteGraphNet) or v1 (route head of a nav TraceNet checkpoint)")
    ap.add_argument("--taus", default="0.4")
    ap.add_argument("--cells", default="exact,s1,s2,s3,s1d30,wrong,none")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dev = "cuda"
    z = pickle.load(open(f"{HOME}/exp_gcar_b123/p10/{a.tok}.pkl", "rb"))
    items, part = z["items"], z["part"]
    if a.n:
        items = items[: a.n]
    if a.kind in ("graph", "x"):
        net = (PN.RouteXNet() if a.kind == "x" else PN.RouteGraphNet()).to(dev)
        net.load_state_dict(torch.load(f"{OUT}/{a.model}/model.pt", map_location=dev))
    elif a.kind in ("loc", "corr"):
        net = (PN.RouteCorrNet() if a.kind == "corr" else PN.RouteLocNet()).to(dev)
        net.load_state_dict(torch.load(f"{OUT}/{a.model}/model.pt", map_location=dev))
    else:
        net = PN.TraceNet(nav=True).to(dev)
        net.load_state_dict(torch.load(f"{OUT}/{a.model}/model.pt", map_location=dev))
    net.eval()
    sdl, rptl = L.cache("sd", part), L.cache("rpts", part)
    taus = [float(t) for t in a.taus.split(",")]
    # lane quality of the traced graph (independent of the map)
    lane = dict(tp_p=0, n_p=0, tp_g=0, n_g=0)
    for it in items:
        gl = L.gt_lane_points(part, it["row"])
        P = np.concatenate([GE.resample(it["xy"][p].astype(np.float64), 0.5) for p in it["pieces"] if len(p) >= 2] or [np.zeros((0, 2))])
        pr = L.lane_pr(P, gl)
        for kk in lane:
            lane[kk] += pr[kk]
    lp, lr = lane["tp_p"] / max(lane["n_p"], 1), lane["tp_g"] / max(lane["n_g"], 1)
    res = {}
    cells = a.cells.split(",")
    acc = {(c, t): dict(n=0, nr=0, succ=0, cov=0.0, prec=0.0, flag_len=0.0, derr=0.0, yerr=0.0, npose=0) for c in cells for t in taus}
    for it in items:
        row = it["row"]
        edges = edges_of(it)
        xy = it["xy"].astype(np.float64)
        n = len(xy)
        if n == 0:
            continue
        tok = torch.from_numpy(it["tok"].astype(np.float32))[None].to(dev)
        xyt = torch.from_numpy(it["xy"])[None].to(dev)
        cpt = torch.from_numpy(it["cp"])[None].to(dev)
        mask = torch.ones(1, n, dtype=torch.bool, device=dev)
        rp = rptl[int(row)]
        for cname in cells:
            sigma, drop, kind, kw = CELLS[cname]
            sd = sdl[row] if kind != "w" else sdl[(int(row) + 397) % len(sdl)]
            if kind != "0" and sd is None:
                continue
            d_true = None
            if kind == "0":
                raster = np.zeros((5, 200, 200), np.float32)
            else:
                sdm, d_true, R_true = L.draw_sd(sd, row, sigma, drop, 1, **kw)
                raster = SL.raster_sd(sdm)
            ras = torch.from_numpy(raster)[None].to(dev)
            with torch.no_grad():
                dh = yh = None
                if a.kind in ("graph", "x"):
                    lg = net(tok, xyt, cpt, ras, mask)
                elif a.kind in ("loc", "corr"):
                    lg, dh, yh = net(tok, xyt, cpt, ras, mask)
                else:
                    lg = net.route_logit(tok, xyt, cpt, ras)
            pr = torch.sigmoid(lg)[0].cpu().numpy()
            for t in taus:
                c = acc[(cname, t)]
                c["n"] += 1
                rt = [(x_, y_, ln) for (x_, y_, ln) in edges if (0.5 * (pr[x_] + pr[y_]) if it["has"][x_] and it["has"][y_] else max(pr[x_] * it["has"][x_], pr[y_] * it["has"][y_])) >= t]
                c["flag_len"] += sum(e[2] for e in rt)
                if dh is not None and kind == "n" and d_true is not None:
                    c["derr"] += float(np.linalg.norm(dh[0].cpu().numpy() - np.asarray(d_true)))
                    c["yerr"] += abs(float(np.degrees(yh[0].cpu().numpy() - np.arctan2(R_true[1, 0], R_true[0, 0]))))
                    c["npose"] += 1
                if kind == "n" and rp is not None and len(rp) > 0:
                    pts = []
                    for x_, y_, ln in rt:
                        m_ = max(int(round(ln)), 1)
                        pts.append(xy[x_][None] + (xy[y_] - xy[x_])[None] * (np.arange(m_ + 1) / m_)[:, None])
                    P = np.concatenate(pts) if pts else np.zeros((0, 2))
                    r = L.route_success(P, rp)
                    if r is not None:
                        c["nr"] += 1
                        c["succ"] += r["succ"]
                        c["cov"] += r["cov"]
                        c["prec"] += r["prec"]
    for (cname, t), c in acc.items():
        nr = max(c["nr"], 1)
        res[f"{cname}/tau{t}"] = dict(n=c["n"], n_route=c["nr"], route_success=c["succ"] / nr, route_cov=c["cov"] / nr, route_prec=c["prec"] / nr, flagged_len_per_row=c["flag_len"] / max(c["n"], 1), pose_err_m=c["derr"] / max(c["npose"], 1), yaw_err_deg=c["yerr"] / max(c["npose"], 1))
        print(f"{cname:7s} tau {t} n {c['n']:4d} route success {c['succ'] / nr:.3f} cov {c['cov'] / nr:.3f} prec {c['prec'] / nr:.3f} flagged length per row {c['flag_len'] / max(c['n'], 1):.1f} m  pose error {c['derr'] / max(c['npose'], 1):.2f} m {c['yerr'] / max(c['npose'], 1):.2f} deg", flush=True)
    res["_lane"] = dict(P=lp, R=lr, F1=2 * lp * lr / max(lp + lr, 1e-9))
    print("traced graph lane P %.3f R %.3f F1 %.3f" % (lp, lr, res["_lane"]["F1"]))
    json.dump(dict(res=res, args=vars(a), n=len(items)), open(f"{OUT}/{a.model}/route2_{a.tag or a.tok}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
