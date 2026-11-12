"""scratch (E283 section 5 'open and not measured', D71): where does the lane GT's junction entry / exit point lie against the painted stop line and the crosswalk of the nuScenes HD map? CPU, VAL 873 frames, no model, no GPU.
Source of the layers: the BEV store of GtGenerator (`stop_line`, `ped_crossing` coverage layers, forward grid x 0..60 m, y -26..26 m, 0.5 m) joined by the sample token; the ends are those of pair_geometry.analyse (clusters of the connector runs of the lane GT l4,
lane frame u = x right, v = y forward). The frames are joined by orientation check first: of the lane GT points that lie inside the BEV grid the share on a `centreline` cell (3 x 3 neighbourhood) under three candidate axis maps; the best one is used and
must reach 0.6.
Measure (per end, in the travel direction t; s = distance along the lane from the end point, + toward the junction for an entry, + toward the junction for an exit means against the travel; the layers are read on the lane line itself, +-0.75 m across):
  stop line: the nearest cell with coverage >= 0.3 within s in [-14, +14] m of the end point; crosswalk the same with ped_crossing. Per kind (entry / exit) and layer: share of the ends with such a cell, median / quartiles of s, share within +-2 m.
PRE-REGISTERED READING (diagnostic by hand, no verdict on a model, no number feeds a decision beyond this one): the stop line is a usable anchor for the end TARGET of an entry (the end head's target would be defined by it) when at least 30 % of the
entries have a stop line within 14 m AND at least 50 % of those have it within +-2 m of the end point; otherwise the end point of the lane GT stays the anchor and the head's target tolerance (2 - 4 m along the lane) stays as written in E283 section 4.
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 bash tools/env/_py14.sh temp/e284/p21_stopline.py"""
import glob
import json
import os
import sys

HEAD = open(os.path.join(os.environ["BFLC_ROOT"], "temp", "e284", "pair_geometry.py")).read().split("# ---- TRAIN set")[0]
exec(compile(HEAD, "pair_geometry_head", "exec"))          # numpy, LD, QE, QT, analyse, ROOT
REPO = os.path.dirname(ROOT)
sys.path.insert(0, REPO)
from GtGenerator import lane_gt  # noqa: E402,F401  (only to make sure the package of this checkout is the one imported)

STORE = None
for sig in sorted(glob.glob(os.path.join(REPO, "data", "autogt", "*")), key=os.path.getmtime, reverse=True):
    if os.path.isdir(sig) and len(glob.glob(os.path.join(sig, "*.npz"))) > 1000:
        STORE = sig
        break
print("BEV store:", STORE, flush=True)
RES, X0, Y0 = 0.5, 0.0, -26.0


def load(tok):
    p = os.path.join(STORE, f"{tok}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k].astype(np.float32) for k in ("stop_line", "ped_crossing", "centreline") if k in z.files}


def to_bev(P, mode):
    P = np.asarray(P, float).reshape(-1, 2)
    if mode == "x=v,y=-u":
        return np.stack([P[:, 1], -P[:, 0]], 1)
    if mode == "x=v,y=u":
        return np.stack([P[:, 1], P[:, 0]], 1)
    return P


def cells(Pb):
    return np.floor((Pb[:, 0] - X0) / RES).astype(int), np.floor((Pb[:, 1] - Y0) / RES).astype(int)


def on(layer, Pb, thr, nb=1):
    ix, iy = cells(Pb)
    ok = (ix >= 0) & (ix < layer.shape[0]) & (iy >= 0) & (iy < layer.shape[1])
    out = np.zeros(len(Pb), bool)
    for dx in range(-nb, nb + 1):
        for dy in range(-nb, nb + 1):
            jx, jy = np.clip(ix + dx, 0, layer.shape[0] - 1), np.clip(iy + dy, 0, layer.shape[1] - 1)
            out |= ok & (layer[jx, jy] >= thr)
    return out, ok


frames = QE.val_frames()
# ---- orientation check on 40 frames
hit = {m: [0, 0] for m in ("x=v,y=-u", "x=v,y=u", "identity")}
n_ok = 0
for f in frames[::22]:
    L = load(f.token)
    if L is None:
        continue
    g = f.lane()
    P = np.concatenate([np.asarray(pc["pts"], float).reshape(-1, 2) for pc in g.get("pieces", []) if len(pc["pts"]) >= 2]) if g.get("pieces") else np.zeros((0, 2))
    if not len(P):
        continue
    n_ok += 1
    for m in hit:
        h, ok = on(L["centreline"], to_bev(P, m), 0.5)
        hit[m][0] += int((h & ok).sum())
        hit[m][1] += int(ok.sum())
    f._lane = f._ref = None
print("orientation check, share of lane GT points inside the BEV grid on a centreline cell:", {m: round(a / max(b, 1), 3) for m, (a, b) in hit.items()}, f"({n_ok} frames)", flush=True)
MODE = max(hit, key=lambda m: hit[m][0] / max(hit[m][1], 1))
assert hit[MODE][0] / max(hit[MODE][1], 1) >= 0.6, hit

# ---- measure
S = np.arange(-14.0, 14.01, 0.5)
res = {(k, ly): [] for k in ("entry", "exit") for ly in ("stop_line", "ped_crossing")}
n_end = {"entry": 0, "exit": 0}
n_in = {"entry": 0, "exit": 0}
nfr = 0
for b, f in enumerate(frames):
    L = load(f.token)
    r = analyse(f.lane())
    f._lane = f._ref = None
    if L is None or r is None:
        continue
    nfr += 1
    for kind, ends in (("entry", r["ent"]), ("exit", r["ext"])):
        for p, t in ends:
            pb = to_bev(p[None], MODE)[0]
            tb = to_bev(t[None], MODE)[0]
            nb_ = np.array([-tb[1], tb[0]])
            n_end[kind] += 1
            if not (2.0 < pb[0] < 56.0 and abs(pb[1]) < 23.0):
                continue
            n_in[kind] += 1
            sg = 1.0 if kind == "entry" else -1.0                      # + = toward the junction
            for ly in ("stop_line", "ped_crossing"):
                best = None
                for s in S:
                    q = pb + sg * s * tb
                    pts = np.stack([q + c * nb_ for c in (-0.75, 0.0, 0.75)])
                    h, ok = on(L[ly], pts, 0.3, nb=0)
                    if h.any():
                        if best is None or abs(s) < abs(best):
                            best = s
                res[(kind, ly)].append(best)
out = dict(mode=MODE, orientation=hit, frames=nfr, ends=n_end, ends_inside_grid=n_in)
for (kind, ly), v in res.items():
    arr = np.array([x for x in v if x is not None], float)
    d = dict(n=len(v), with_layer=int(len(arr)), share=float(len(arr) / max(len(v), 1)))
    if len(arr):
        d.update(median=float(np.median(arr)), q25=float(np.percentile(arr, 25)), q75=float(np.percentile(arr, 75)), within2=float((np.abs(arr) <= 2.0).mean()),
                 upstream_of_point=float((arr < -0.25).mean()), inside_of_point=float((arr > 0.25).mean()))
    out[f"{kind}/{ly}"] = d
json.dump(out, open(os.path.join(ROOT, "temp", "e284", "p21_stopline.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
sl = out["entry/stop_line"]
print("READING: entry stop line a usable anchor:", sl["share"] >= 0.30 and sl.get("within2", 0) >= 0.5, f"(share with a stop line {sl['share']:.2f}, within +-2 m {sl.get('within2', 0):.2f})")
