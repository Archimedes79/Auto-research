"""Loop 16b: why is the route success of the sparse builder lower?  Per frame, exact map, tau 0.5: route coverage by ALL edges of the graph (an upper bound: nothing is flagged wrongly) against the coverage by the FLAGGED edges,
and the success of both. Separates a recall problem of the graph from a flagging problem of the route head.
Usage (gcar, GPU): p16_route_diag.py --tok tok_p16_f2h_val_clean --model p16_f2h_route_s0 --name champion"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p8_lib as L  # noqa: E402
import p10_net as PN  # noqa: E402
import sd_lib as SL  # noqa: E402
import p10_route_eval2 as RE  # noqa: E402

HOME = os.path.expanduser("~")
OUT = L.OUT


def pts_of(xy, edges):
    pts = []
    for a, b, ln in edges:
        m_ = max(int(round(ln)), 1)
        pts.append(xy[a][None] + (xy[b] - xy[a])[None] * (np.arange(m_ + 1) / m_)[:, None])
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--kind", default="x")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--split_part", default="")
    a = ap.parse_args()
    dev = "cuda"
    z = pickle.load(open(f"{HOME}/exp_gcar_b123/p10/{a.tok}.pkl", "rb"))
    items, part = z["items"], z["part"]
    net = (PN.RouteXNet() if a.kind == "x" else PN.RouteGraphNet()).to(dev)
    net.load_state_dict(torch.load(f"{OUT}/{a.model}/model.pt", map_location=dev))
    net.eval()
    sdl, rptl = L.cache("sd", part), L.cache("rpts", part)
    rows = []
    for it in items:
        row = it["row"]
        xy = it["xy"].astype(np.float64)
        n = len(xy)
        rp = rptl[int(row)]
        sd = sdl[row]
        if n == 0 or rp is None or len(rp) == 0 or sd is None:
            continue
        edges = RE.edges_of(it)
        tok = torch.from_numpy(it["tok"].astype(np.float32))[None].to(dev)
        xyt = torch.from_numpy(it["xy"])[None].to(dev)
        cpt = torch.from_numpy(it["cp"])[None].to(dev)
        mask = torch.ones(1, n, dtype=torch.bool, device=dev)
        sdm, d_true, R_true = L.draw_sd(sd, row, 0.0, 0.0, 1)
        ras = torch.from_numpy(SL.raster_sd(sdm))[None].to(dev)
        with torch.no_grad():
            pr = torch.sigmoid(net(tok, xyt, cpt, ras, mask))[0].cpu().numpy()
        rt = [(x_, y_, ln) for (x_, y_, ln) in edges if (0.5 * (pr[x_] + pr[y_]) if it["has"][x_] and it["has"][y_] else max(pr[x_] * it["has"][x_], pr[y_] * it["has"][y_])) >= a.tau]
        r_fl = L.route_success(pts_of(xy, rt), rp)
        r_all = L.route_success(pts_of(xy, edges), rp)
        if r_fl is None or r_all is None:
            continue
        rows.append(dict(row=int(row), succ=float(r_fl["succ"]), cov=float(r_fl["cov"]), prec=float(r_fl["prec"]), succ_all=float(r_all["succ"]), cov_all=float(r_all["cov"]), prec_all=float(r_all["prec"])))
    m = lambda k: float(np.mean([r[k] for r in rows]))
    out = dict(name=a.name, n=len(rows), success_flagged=m("succ"), cov_flagged=m("cov"), prec_flagged=m("prec"), success_if_all_edges_flagged=m("succ_all"), cov_all_edges=m("cov_all"), prec_all_edges=m("prec_all"),
               frames_cov_all_below_0p8=float(np.mean([r["cov_all"] < 0.8 for r in rows])), succ_given_cov_all_ok=float(np.mean([r["succ"] for r in rows if r["cov_all"] >= 0.8])) if any(r["cov_all"] >= 0.8 for r in rows) else None,
               frames_flag_fail_with_cov_all_ok=float(np.mean([(r["succ"] < 0.5) and (r["cov_all"] >= 0.8) for r in rows])))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)
    json.dump(dict(summary=out, rows=rows), open(f"{OUT}/e123_ladder/diag/routediag_{a.name}.json", "w"))


if __name__ == "__main__":
    main()
