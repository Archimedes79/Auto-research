"""scratch (Mirko 2026-11-10, D71: the junction points - entries (stop) AND exits (exit) - are to lie as LATE as possible, but where the network can see them).
Where do the delivered ends lie ALONG THE LANE against the map's point, and what is visible / painted around the map's point? CPU, VAL 873, baseline B + hard close-node step 1.5 m (E275 read-out), both seeds.
GT entry = first point of a connector run, GT exit = last point of a connector run (runs of >= 3 points, as ends_quality.py); the lane direction t = the tangent there (travel order).
 1. OFFSET   for every GT entry / exit: the delivered kept node nearest in |s| inside the corridor |s| <= 8 m along the lane, |lateral| <= 1.5 m, heading within 30 degrees (any kept node, and the node with the stop / exit flag >= 0.5).
             Reported TOWARD THE JUNCTION: + = the node lies nearer the junction than the map's point (later for an entry, earlier along the travel for an exit), - = farther from the junction (the network puts the end earlier / further out).
             Shares: no node in the corridor; toward <= -2 m; |.| < 1 m; >= +2 m; quantiles of the offset.
 2. SEEN     seed-free, from the lane GT: along the lane at offsets s = -12 ... +8 m from the map's point (toward the junction = +): the share of events whose lane point at that offset is (a) seen by a camera and not hidden by a box (l4 `vis`),
             (b) within 2.5 m of a painted lane / road divider, (c) within 2.5 m of the drivable-area boundary. Only events with the lane points available at that offset (the run starts / ends inside its piece).
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 bash tools/env/_py14.sh temp/e284/p17_end_offset.py"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as QR  # noqa: E402
import qtargets as QT  # noqa: E402

SPEC = QR.parse("close:1.5@0.5")
FI = QT.FI
frames = QE.val_frames()
S_OFFS = np.arange(-12.0, 8.5, 2.0)                     # toward the junction = +
NEAR = 2.5


def line_points(g, key):
    L = g.get("lines", {})
    L = L.get(key, []) if isinstance(L, dict) else []
    pts = [np.asarray(x["pts"], float).reshape(-1, 2) for x in L if len(x.get("pts", []))]
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def runs_of(g):
    """-> list of (kind 'ent'/'ext', point, tangent, piece, index k) for every connector run of >= 3 points"""
    out = []
    for pc in g.get("pieces", []):
        q = np.asarray(pc["pts"], float).reshape(-1, 2)
        if len(q) < 4:
            continue
        tan = QT._tangents(q)
        conn = np.zeros(len(q), bool)
        for s in pc["segments"]:
            if s["kind"] == "connector":
                conn[s["i0"]: s["i1"] + 1] = True
        k = 0
        while k < len(q):
            if conn[k]:
                j = k
                while j + 1 < len(q) and conn[j + 1]:
                    j += 1
                if j - k + 1 >= 3:
                    if (np.abs(q[k]) < 48).all():
                        out.append(("ent", q[k], tan[k], pc, k))
                    if (np.abs(q[j]) < 48).all():
                        out.append(("ext", q[j], tan[j], pc, j))
                k = j + 1
            else:
                k += 1
    return out


def dedupe(items, tol_pos=1.5, tol_ang=40.0):
    reps = []
    for it in items:
        for r in reps:
            if r[0] == it[0] and np.linalg.norm(it[1] - r[1]) <= tol_pos and np.degrees(np.arccos(np.clip(it[2] @ r[2], -1, 1))) <= tol_ang:
                break
        else:
            reps.append(it)
    return reps


def summarise(v, n_events):
    v = np.asarray(v, float)
    if not len(v):
        return dict(events=n_events, with_node=0)
    return dict(events=n_events, with_node=int(len(v)), share_with_node=len(v) / n_events, median=float(np.median(v)), q10=float(np.quantile(v, 0.1)), q25=float(np.quantile(v, 0.25)), q75=float(np.quantile(v, 0.75)), q90=float(np.quantile(v, 0.9)),
                share_le_m2=float((v <= -2).mean()), share_within_1=float((np.abs(v) < 1).mean()), share_ge_p2=float((v >= 2).mean()), share_le_m1=float((v <= -1).mean()), share_ge_p1=float((v >= 1).mean()))


res = {}
# ------------------------------------------------------------------ 2. SEEN (seed free)
seen = {k: {s: [0, 0, 0, 0] for s in S_OFFS} for k in ("ent", "ext")}   # per offset: [n available, vis, near divider, near boundary]
ev_vis = {"stop": [0, 0], "exit": [0, 0]}
n_events = {"ent": 0, "ext": 0}
for n, f in enumerate(frames):
    g = f.lane()
    ru = dedupe(runs_of(g))
    if not ru:
        f._lane = f._ref = None
        continue
    div = line_points(g, "lane_divider")
    rdiv = line_points(g, "road_divider")
    allpaint = np.concatenate([div, rdiv]) if len(div) + len(rdiv) else np.zeros((0, 2))
    bnd = line_points(g, "boundary")
    tp = cKDTree(allpaint) if len(allpaint) else None
    tb = cKDTree(bnd) if len(bnd) else None
    for kind, p, t, pc, k in ru:
        n_events[kind] += 1
        q = np.asarray(pc["pts"], float).reshape(-1, 2)
        vis = np.asarray(pc["vis"], bool) if "vis" in pc else None
        for s in S_OFFS:
            # entry: toward the junction = along the travel (index +); exit: toward the junction = against the travel (index -)
            idx = k + int(round((s if kind == "ent" else -s) / 0.5))
            if idx < 0 or idx >= len(q):
                continue
            cell = seen[kind][s]
            cell[0] += 1
            if vis is not None and vis[idx]:
                cell[1] += 1
            if tp is not None and tp.query(q[idx])[0] <= NEAR:
                cell[2] += 1
            if tb is not None and tb.query(q[idx])[0] <= NEAR:
                cell[3] += 1
    for e in g.get("events", []):
        if e.get("type") in ("stop", "exit") and (np.abs(np.asarray(e["pos"], float)) < 48).all() and "vis" in e:
            ev_vis[e["type"]][0] += 1
            ev_vis[e["type"]][1] += int(bool(e["vis"]))
    f._lane = f._ref = None
    if (n + 1) % 200 == 0:
        print(f"  seen pass {n + 1}/{len(frames)}", flush=True)
res["seen"] = {kind: {f"{s:+.0f} m": dict(n=c[0], vis=(c[1] / c[0] if c[0] else None), near_paint=(c[2] / c[0] if c[0] else None), near_boundary=(c[3] / c[0] if c[0] else None)) for s, c in seen[kind].items()} for kind in seen}
res["seen"]["events"] = n_events
res["seen"]["event_vis"] = {k: dict(n=v[0], vis=(v[1] / v[0] if v[0] else None)) for k, v in ev_vis.items()}
print("SEEN (toward the junction = +; share of events whose lane point at that offset is camera-visible / within 2.5 m of paint / of the boundary):")
for kind in ("ent", "ext"):
    print(f"  {kind}: events {n_events[kind]}")
    for s in S_OFFS:
        c = seen[kind][s]
        if c[0]:
            print(f"    s {s:+5.0f} m  n {c[0]:5d}  vis {c[1] / c[0]:.2f}  paint {c[2] / c[0]:.2f}  boundary {c[3] / c[0]:.2f}")
print("  l4 event vis (the map's stop / exit point seen by a camera):", {k: (round(v[1] / v[0], 3) if v[0] else None, v[0]) for k, v in ev_vis.items()})

# ------------------------------------------------------------------ 1. OFFSET (per seed)
for SEED in (0, 1):
    dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
    off = {(kind, pool): [] for kind in ("ent", "ext") for pool in ("any", "flag")}
    cnt = {kind: 0 for kind in ("ent", "ext")}
    for n, (f, d) in enumerate(zip(frames, dump)):
        g = f.lane()
        ru = dedupe(runs_of(g))
        if not ru:
            f._lane = f._ref = None
            continue
        pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                             links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), SPEC)
        graph, bk = QE.readout(pred, 0.5, 0.5)
        kp = np.flatnonzero(bk["keep"])
        pos = d["pos"]
        u = d["dirv"] / (np.linalg.norm(d["dirv"], axis=1, keepdims=True) + 1e-9)
        fl = d["flags"]
        pools = {"any": kp, "stop": kp[fl[kp, FI["stop"]] >= 0.5], "exit": kp[fl[kp, FI["exit"]] >= 0.5]}
        for kind, p, t, pc, k in ru:
            cnt[kind] += 1
            toward = 1.0 if kind == "ent" else -1.0
            nrm = np.array([-t[1], t[0]])
            for pool_name, pool in (("any", pools["any"]), ("flag", pools["stop" if kind == "ent" else "exit"])):
                if not len(pool):
                    continue
                dp = pos[pool] - p
                s_al = dp @ t
                lat = dp @ nrm
                ok = (np.abs(s_al) <= 8.0) & (np.abs(lat) <= 1.5) & (u[pool] @ t >= np.cos(np.radians(30.0)))
                if ok.any():
                    i = np.flatnonzero(ok)[np.argmin(np.abs(s_al[ok]))]
                    off[(kind, pool_name)].append(toward * s_al[i])
        f._lane = f._ref = None
        if (n + 1) % 200 == 0:
            print(f"  seed {SEED} offset pass {n + 1}/{len(frames)}", flush=True)
    res[f"seed {SEED}"] = {f"{kind}_{pool}": summarise(off[(kind, pool)], cnt[kind]) for kind in ("ent", "ext") for pool in ("any", "flag")}
    print(f"OFFSET seed {SEED} (toward the junction, metres; + = the delivered end is nearer the junction than the map's point):")
    for kind in ("ent", "ext"):
        for pool in ("any", "flag"):
            r = res[f"seed {SEED}"][f"{kind}_{pool}"]
            if r.get("with_node"):
                print(f"  {kind} {pool:4s}: events {r['events']}, node in corridor {100 * r['share_with_node']:.1f} %, median {r['median']:+.2f}, q25 {r['q25']:+.2f}, q75 {r['q75']:+.2f}, <= -2 m {100 * r['share_le_m2']:.1f} %, |.| < 1 m {100 * r['share_within_1']:.1f} %, >= +2 m {100 * r['share_ge_p2']:.1f} %")
json.dump(res, open(os.path.join(ROOT, "temp", "e284", "p17_end_offset.json"), "w"), indent=1)
print("done")
