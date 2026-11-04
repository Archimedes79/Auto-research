"""object_shares versus an independent per-sample reference (shapely 2D clip + z interval, own projection)."""
import os, sys, math, pickle
import numpy as np
from shapely.geometry import LineString, Polygon, Point
from autogt import coverage as C
from autogt.cli import load_keyframe
from autogt.grid import GridSpec
sys.path.insert(0, "bev_fusion/temp/critic_coverage")
grid = GridSpec()
rp = next(iter(pickle.load(open("bev_fusion/temp/critic_coverage/dump_new.pkl", "rb")).values()))["_radar_poses"]


def ref_blocked(cam, p, prism):
    fp, z_lo, z_hi = prism
    poly = Polygon(fp)
    if poly.covers(Point(p[:2])):
        return False
    inter = LineString([cam[:2], p[:2]]).intersection(poly)
    if inter.is_empty or inter.geom_type == "Point":
        return False
    d = p - cam; L2 = float(d[:2] @ d[:2]); ts = []
    for gm in ([inter] if inter.geom_type == "LineString" else list(inter.geoms)):
        if gm.geom_type == "LineString":
            ts += [((np.array(q) - cam[:2]) @ d[:2]) / L2 for q in gm.coords]
    t0, t1 = min(ts), max(ts)
    if t1 - t0 < 1e-9:
        return False
    za, zb = cam[2] + t0 * d[2], cam[2] + t1 * d[2]
    return max(za, zb) >= z_lo and min(za, zb) <= z_hi


d = "data/autogt/15efbde2d1"
files = sorted(f for f in os.listdir(d) if f.endswith(".npz") and ".tmp" not in f)
rng = np.random.default_rng(21)
nb = nbad = 0; maxdiff = 0.0; skipdiff = 0
for i in rng.choice(len(files), 40, replace=False):
    g = load_keyframe(d, files[i][:-4]); m = g["meta"]; K = m["K"]; T = m["T_c2e"]; cam = T[:3, 3]
    boxes = m["boxes"]; hiding = [b for b in boxes if b["observed"]]; prisms = [C.prism_of(b) for b in hiding]
    place = {id(b): n for n, b in enumerate(hiding)}
    for b in boxes:
        sh = C.object_shares(b, place.get(id(b)), prisms, cam, K, T, (1600, 900), rp, grid)
        sh2 = C.object_shares(b, None, prisms, cam, K, T, (1600, 900), rp, grid)       # own prism not skipped
        skipdiff += sh != sh2
        fp, lo, hi = C.prism_of(b)
        xy = C.footprint_samples(fp)
        ok = C._on_grid(xy, grid)
        if not ok.any():
            assert sh == {"camera": 0.0, "radar": 0.0, "fusion": 0.0}; continue
        # reference only on a subsample of the samples to bound the run time
        sel = np.nonzero(ok)[0]
        if len(sel) > 60:
            sel = rng.choice(sel, 60, replace=False)
        seen = []
        for k in sel:
            vis_any = False
            for fr in C.FRACTIONS:
                z = lo + fr * (hi - lo)
                p = np.array([xy[k, 0], xy[k, 1], z])
                inimg = bool(C.camera_cells(np.array([p[0]]), np.array([p[1]]), K, T, (1600, 900), z=np.array([z]))[0])
                if inimg and not any(ref_blocked(cam, p, pr) for j, pr in enumerate(prisms) if j != place.get(id(b))):
                    vis_any = True; break
            seen.append(vis_any)
        ref_share_ok = np.mean(seen)                                   # share of the on-grid samples that are seen
        got = sh["camera"] * len(xy) / ok.sum()                         # share among the on-grid samples
        # compare with the subsampled estimate: tolerance by sampling error
        nb += 1; maxdiff = max(maxdiff, abs(ref_share_ok - got))
        tol = 3 * math.sqrt(max(got * (1 - got), 1e-4) / len(sel)) + 0.02
        if abs(ref_share_ok - got) > max(tol, 0.12):
            nbad += 1; print("BAD", b["group"], b["size"], got, ref_share_ok)
print("boxes compared", nb, "beyond tolerance", nbad, "max |diff| (subsampled ref)", round(maxdiff, 3), "boxes where the skip changed the result:", skipdiff)
