"""Loop 17 / E210 (internal E143): the map-only baseline. The SD map alone, drawn as a lane graph (every road polyline with its lane count -> lane centre lines at 3.5 m spacing, cropped to +-48 m, no connections), scored like a perception graph against the ground-truth lane graph
on the failure-half frames of the validation series, for a map without error and with a scene-constant pose error of 1 and 2 m; the route success of the SD route polyline itself. No perception, no sensor: what a navigation map alone gives (it is derived from the lane map, so an upper bound).
Usage (gcar): p17_maponly.py"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sd_lib as RT11  # noqa: E402  (scene_pose_error, sd_raster: moved from the loop-11 script)
import p16_common as C  # noqa: E402
import p17_eval as EV  # noqa: E402
import p8_lib as L  # noqa: E402

E17 = EV.E17
ds = EV.SeriesData("dark_single", with_prev=False)
meta = ds.meta
P = ds.P17
sdl, rptl = L.cache("sd", "val"), L.cache("rpts", "val")
gts_all = ds.gt
rows = [int(r) for r in meta["rows"]]
fail = np.asarray(meta["phase"]) == 1


def densify(p, step=1.0):
    p = np.asarray(p, np.float64)
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, s[-1] + 1e-9, step)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def lanes_of(poly, cnt, spacing=3.5):
    p = densify(poly)
    if len(p) < 3:
        return []
    tan = np.gradient(p, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
    c_ = np.asarray(cnt, np.float64).reshape(-1)  # lane count per point of the road (or one number)
    n = int(max(1, round(float(np.median(c_)) if c_.size else 1)))
    out = []
    for k in range(n):
        q = p + nrm * ((k - (n - 1) / 2.0) * spacing)
        keep = (np.abs(q[:, 0]) < 48) & (np.abs(q[:, 1]) < 48)
        if keep.sum() >= 2:
            out.append(q[keep])
    return out


def map_graph(sdm, R, d):
    pts = []
    for poly, cnt in sdm["roads"]:
        q = np.asarray(poly, np.float64)
        if len(q) < 2:
            continue
        if R is not None:
            q = q @ R.T + d
        pts += lanes_of(q, cnt)
    return dict(pts=pts, edges=[], nodes=[list(range(len(p))) for p in pts])


def route_pts(sdm, R, d):
    out = []
    for poly, cnt in sdm["route"]:
        q = np.asarray(poly, np.float64)
        if len(q) < 2:
            continue
        if R is not None:
            q = q @ R.T + d
        out.append(densify(q, 0.5))
    return np.concatenate(out) if out else np.zeros((0, 2))


res = {}
for sigma in (0.0, 1.0, 2.0):
    graphs, gts, succ = [], [], []
    for i, row in enumerate(rows):
        if not fail[i]:
            continue
        sd = sdl[row]
        if sd is None:
            continue
        sdm, dd, R = L.draw_sd(sd, row, 0.0, 0.0, 1) if sigma == 0 else L.draw_sd(sd, row, sigma, 0.0, 1)
        if sigma > 0:  # scene-constant error as in the other experiments
            w, th = RT11.scene_pose_error(int(P["scene"][row]), sigma)
            dR = P["R"][row].T @ w
            import assoc_lib as AL

            Rr, dd_ = AL.rot2(th), dR
            sdm, _, _ = L.draw_sd(sd, row, 0.0, 0.0, 1)
        else:
            Rr, dd_ = None, None
        graphs.append(map_graph(sdm, Rr, dd_))
        gts.append(gts_all[row])
        rp = rptl[row]
        r = L.route_success(route_pts(sdm, Rr, dd_), rp)
        if r is not None:
            succ.append(r["succ"])
    c = C.card(graphs, gts)
    res[f"sigma{sigma:g}"] = dict(n=len(graphs), route_n=len(succ), route_success=float(np.mean(succ)) if succ else float("nan"), **{k: float(c[k]) for k in ("S", "F1@1.0", "F1@0.5", "J", "vert100", "false_len_frac", "lanes_found", "R@1.0", "P@1.0")})
    print(sigma, {k: round(v, 3) for k, v in res[f"sigma{sigma:g}"].items()}, flush=True)
json.dump(res, open(f"{E17}/e143_maponly.json", "w"), indent=1)
