"""scratch (analysis, no verdict): the PRECISION side of the lane_divider output. The 246 frames of the recall test (rp_out/<i>.npz = dt-0 probability maps of runs/t20_base_s0, memory empty,
CPU) are read again; no inference. Per PREDICTED cell (p > 0.5, the metric's rule; inside the GT camera (fov) mask; 5-30 m ahead; the cell centre projected into the front image with its
whole +-1 m profile inside the image):
  TRUE  = a GT lane_divider cell within 0.5 m (the metric's tolerance disc, evalsuite.comparison.maps.dilate, 1 cell)
  FALSE = none within 0.5 m;  FALSE-near = 0.5-1.0 m from a GT lane cell (disc of 2 cells), FALSE-far = more than 1.0 m
1. stripe flag (paint_vs_label.profiles + contrast_hits: contrast >= 0.25 within +-0.5 m, and +-0.2 m) at the cell centre, across the line. The line direction is the principal axis of the
   predicted cells (p > 0.5) within 1.5 m of the cell (needs >= 4 cells and an eigenvalue ratio >= 3 = a 'clear' axis; otherwise the ego-forward axis is used and the cell is marked unclear).
   Control = the same cells moved 1.75 m sideways (both sides pooled). Scene-level bootstrap (2000) for the 95 % intervals.
2. position against the other GT layers of the same cached frame (signature 6a7aa7864a): road_divider, ped_crossing, stop_line, walkway boundary / inside, drivable area, carpark, GT centreline,
   box footprint (instance >= 0), occluded; the same shares for TRUE cells, for all fov cells of the range (base rate) and for the GT lane cells.
3. junction: nuScenes map road_segment polygons with is_intersection, rasterised into the ego grid (nuscenes.map_expansion, read only; the pose is meta['T_e2g']).
Output: temp/overview/precision_paint_result.txt / .json, temp/overview/precision_paint_cells.npz (per-cell table), pp_contact data for the picture script.
python temp/overview/pp_precision.py"""
import json
import os
import pickle
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch"), os.path.join(ROOT, "temp", "lane_tests")]
import paint_vs_label as P  # noqa: E402
from evalsuite.comparison import maps as M  # noqa: E402

RES, X0, Y0, NX, NY = 0.5, 0.0, -26.0, 120, 104
XC = (X0 + (np.arange(NX) + 0.5) * RES)[:, None] * np.ones((1, NY))
YC = np.ones((NX, 1)) * (Y0 + (np.arange(NY) + 0.5) * RES)[None, :]
RANGES = ((5.0, 12.0), (12.0, 20.0), (20.0, 30.0))
SHIFT = 1.75
NB = 2000
rng = np.random.RandomState(0)
FROOT = os.path.expanduser("~/data/nuscenes_full")


def axes_of(pc, radius=1.5):
    """principal axis of the predicted cells within `radius` m of every cell of pc (N, 2): (unit axis (N, 2), clear (N,), count (N,), ratio (N,))"""
    N = len(pc)
    d = pc[:, None, :] - pc[None, :, :]
    w = (d ** 2).sum(-1) <= radius ** 2 + 1e-9
    n = w.sum(1)
    mu = (w[:, :, None] * pc[None, :, :]).sum(1) / n[:, None]
    dc = pc[None, :, :] - mu[:, None, :]
    C = np.einsum("ij,ijk,ijl->ikl", w.astype(float), dc, dc) / n[:, None, None]
    vals, vecs = np.linalg.eigh(C)
    ax = vecs[:, :, 1]
    ratio = vals[:, 1] / np.maximum(vals[:, 0], 0.02)               # 0.02 = the variance of a cell's own extent (0.5 m / sqrt(12))^2
    clear = (n >= 4) & (ratio >= 3.0)
    ax = np.where(clear[:, None], ax, np.array([1.0, 0.0])[None, :])
    return ax, clear, n, ratio


def centre_valid(meta, shape):
    """cells whose centre (ego x, y, z = 0) projects into the image the way paint_vs_label.sample_gray requires"""
    K, T = np.asarray(meta["K"], float), np.asarray(meta["T_c2e"], float)
    xy = np.c_[XC.ravel(), YC.ravel(), np.zeros(XC.size), np.ones(XC.size)]
    Pc = xy @ np.linalg.inv(T).T
    z = Pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = K[0, 0] * Pc[:, 0] / z + K[0, 2], K[1, 1] * Pc[:, 1] / z + K[1, 2]
    ok = (z > 1.0) & (u > 2) & (u < shape[1] - 3) & (v > 2) & (v < shape[0] - 3)
    return ok.reshape(NX, NY)


def walkway_boundary(wk):
    pad = np.pad(wk, 1, mode="edge")
    inner = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return wk & ~inner


class Maps:
    """intersection polygons of the nuScenes map of a scene's location (nuscenes.map_expansion, read only)"""

    def __init__(self):
        self.scene_loc = {}
        scenes = {s["token"]: s for s in json.load(open(os.path.join(FROOT, "v1.0-trainval", "scene.json")))}
        logs = {s["token"]: s for s in json.load(open(os.path.join(FROOT, "v1.0-trainval", "log.json")))}
        for t, s in scenes.items():
            self.scene_loc[t] = logs[s["log_token"]]["location"]
        self.polys = {}

    def intersections(self, loc):
        if loc not in self.polys:
            from nuscenes.map_expansion.map_api import NuScenesMap
            nm = NuScenesMap(dataroot=FROOT, map_name=loc)
            out = []
            for r in nm.road_segment:
                if r.get("is_intersection"):
                    pg = nm.extract_polygon(r["polygon_token"])
                    out.append((pg.bounds, pg))
            self.polys[loc] = out
            del nm
        return self.polys[loc]

    def raster(self, scene_token, T_e2g):
        """(NX, NY) bool: cell centre inside an intersection road segment"""
        polys = self.intersections(self.scene_loc[scene_token])
        T = np.asarray(T_e2g, float)
        ex, ey = T[0, 3], T[1, 3]
        Tinv = np.linalg.inv(T)
        ss = 4
        img = Image.new("L", (NY * ss, NX * ss), 0)
        dr = ImageDraw.Draw(img)

        def to_px(coords):
            c = np.asarray(coords, float)[:, :2]
            e = np.c_[c, np.zeros(len(c)), np.ones(len(c))] @ Tinv.T
            return [(float((y - Y0) / RES * ss), float((x - X0) / RES * ss)) for x, y in zip(e[:, 0], e[:, 1])]
        for (x0, y0, x1, y1), pg in polys:
            if x1 < ex - 90 or x0 > ex + 90 or y1 < ey - 90 or y0 > ey + 90:
                continue
            geoms = [pg] if pg.geom_type == "Polygon" else list(pg.geoms)
            for g in geoms:
                dr.polygon(to_px(g.exterior.coords), fill=1)
            for g in geoms:
                for it in g.interiors:
                    dr.polygon(to_px(it.coords), fill=0)
        a = np.asarray(img, bool)
        return a[ss // 2::ss, ss // 2::ss][:NX, :NY]


def main():
    frames = json.load(open(os.path.join(HERE, "rp_frames.json")))
    done = [f for f in frames if os.path.exists(os.path.join(HERE, "rp_out", f"{f['i']}.npz"))]
    scene_ids = {s: k for k, s in enumerate(sorted({f["scene"] for f in frames}))}
    print(f"{len(done)} frames with a stored output", flush=True)
    maps = Maps()
    cells = {k: [] for k in ("fi", "scene", "x", "y", "dist", "clear", "okr", "hit05", "hit02", "c_ok", "c_hit05", "c_hit02", "inter", "occ", "box", "dr", "wk",
                              "cp", "feat", "ci", "cj", "ratio")}
    base = {"n": 0, "feat": None, "inter": 0, "n_inter_known": 0}
    gtl = {"n": 0, "feat": None, "inter": 0}
    prec = {"all": np.zeros(2), "0-20": np.zeros(2), "20-40": np.zeros(2), "40-60": np.zeros(2)}          # [hits, cells] of the metric's logic (fov mask)
    FEATS = ["rd05", "rd10", "pc05", "pc10", "sl05", "sl10", "wb05", "wb10", "cl05", "cl10", "oth10", "oth_cl10", "unexpl"]
    for k, f in enumerate(done):
        i = f["i"]
        g = np.load(os.path.join(ROOT, "data", "cache", "gt", f["file"]), allow_pickle=True)["gt"].item()["cur"]
        meta = g["meta"]
        prob = np.load(os.path.join(HERE, "rp_out", f"{i}.npz"))["lane_divider"].astype(np.float32)
        Pm = prob > 0.5
        cam = np.asarray(g["camera"], np.float32) > 0
        Gl = np.asarray(g["lane_divider"], np.float32) > 0.5
        D05 = M.dilate(Gl, 1)
        dcell = ndi.distance_transform_edt(~Gl) if Gl.any() else np.full(Gl.shape, 99.0)
        for nm, lo, hi in (("all", 0, 1e9), ("0-20", 0, 20), ("20-40", 20, 40), ("40-60", 40, 60)):
            m = Pm & cam & (XC >= lo) & (XC < hi)
            prec[nm] += [(m & D05).sum(), m.sum()]
        # other GT layers
        layer = lambda name: np.asarray(g[name], np.float32) > 0.5            # noqa: E731
        rd, pc_, sl, wk, dr_, cp_, cl = (layer(n) for n in ("road_divider", "ped_crossing", "stop_line", "walkway", "drivable_area", "carpark_area", "centreline"))
        wb = walkway_boundary(wk)
        box = np.asarray(g["instance"]) >= 0
        occ = np.asarray(g["occluded"], np.float32) > 0
        F = {"rd05": M.dilate(rd, 1), "rd10": M.dilate(rd, 2), "pc05": M.dilate(pc_, 1), "pc10": M.dilate(pc_, 2), "sl05": M.dilate(sl, 1), "sl10": M.dilate(sl, 2),
             "wb05": M.dilate(wb, 1), "wb10": M.dilate(wb, 2), "cl05": M.dilate(cl, 1), "cl10": M.dilate(cl, 2)}
        F["oth10"] = F["rd10"] | F["pc10"] | F["sl10"] | F["wb10"]
        F["oth_cl10"] = F["oth10"] | F["cl10"]
        F["unexpl"] = ~(F["oth_cl10"] | box | occ)
        inter = maps.raster(meta["scene"], meta["T_e2g"])
        im = np.asarray(Image.open(os.path.join(FROOT, meta["cam_file"])).convert("L"), np.float32)
        cv = centre_valid(meta, im.shape)
        rng_ = (XC >= 5) & (XC < 30)
        # base rates over the fov cells of the range that project into the image; GT lane cells likewise
        for acc, sel in ((base, cam & rng_ & cv), (gtl, Gl & cam & rng_ & cv)):
            fv = np.array([F[n][sel].sum() for n in FEATS] + [box[sel].sum(), occ[sel].sum(), dr_[sel].sum(), wk[sel].sum(), cp_[sel].sum()], float)
            acc["feat"] = fv if acc["feat"] is None else acc["feat"] + fv
            acc["n"] += int(sel.sum())
            acc["inter"] += int(inter[sel].sum())
        # predicted cells
        allp = np.argwhere(Pm)
        if len(allp) == 0:
            continue
        pcs = np.c_[XC[Pm], YC[Pm]]
        ax_all, clear_all, n_all, ratio_all = axes_of(pcs)
        sel = Pm & cam & rng_ & cv
        if not sel.any():
            continue
        ci, cj = np.nonzero(sel)
        # axis of those cells: index into the predicted-cells list (np.nonzero order of Pm is row-major, the same as XC[Pm])
        order = np.cumsum(Pm.ravel()) - 1
        idx = order.reshape(NX, NY)[ci, cj]
        ax, clear, ratio = ax_all[idx], clear_all[idx], ratio_all[idx]
        pts = np.c_[XC[ci, cj], YC[ci, cj]]
        real = P.profiles(im, meta, pts, ax, 0.0)
        okr = ~np.isnan(real).any(1)
        hr = P.contrast_hits(np.where(okr[:, None], real, 0.0))
        cok, ch5, ch2 = [], [], []
        for s in (SHIFT, -SHIFT):
            pr = P.profiles(im, meta, pts, ax, s)
            ok = ~np.isnan(pr).any(1)
            h = P.contrast_hits(np.where(ok[:, None], pr, 0.0))
            cok.append(ok)
            ch5.append(h["hit05"] & ok)
            ch2.append(h["hit02"] & ok)
        n = len(ci)
        cells["fi"].append(np.full(n, i)); cells["scene"].append(np.full(n, scene_ids[f["scene"]]))
        cells["x"].append(pts[:, 0]); cells["y"].append(pts[:, 1]); cells["dist"].append(dcell[ci, cj] * RES)
        cells["clear"].append(clear); cells["ratio"].append(ratio); cells["okr"].append(okr)
        cells["hit05"].append(hr["hit05"] & okr); cells["hit02"].append(hr["hit02"] & okr)
        cells["c_ok"].append(np.stack(cok)); cells["c_hit05"].append(np.stack(ch5)); cells["c_hit02"].append(np.stack(ch2))
        cells["inter"].append(inter[ci, cj]); cells["occ"].append(occ[ci, cj]); cells["box"].append(box[ci, cj])
        cells["dr"].append(dr_[ci, cj]); cells["wk"].append(wk[ci, cj]); cells["cp"].append(cp_[ci, cj])
        cells["feat"].append(np.stack([F[nm][ci, cj] for nm in FEATS], 1))
        cells["ci"].append(ci); cells["cj"].append(cj)
        if (k + 1) % 40 == 0:
            print(f"  {k + 1}/{len(done)}", flush=True)
    A = {}
    for key, v in cells.items():
        A[key] = np.concatenate(v, axis=-1 if key in ("c_ok", "c_hit05", "c_hit02") else 0)
    np.savez(os.path.join(HERE, "precision_paint_cells.npz"), **A, FEATS=np.array(FEATS))
    pickle.dump({"prec": prec, "base": base, "gtl": gtl, "FEATS": FEATS, "frames": [f["i"] for f in done]}, open(os.path.join(HERE, "pp_aux.pkl"), "wb"))
    print("cells:", len(A["x"]), flush=True)


if __name__ == "__main__":
    main()
