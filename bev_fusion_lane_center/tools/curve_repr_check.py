"""How well do nodes describe a lane as a curve? CPU only, no training, on the GT lane paths of VAL (`lane_gt_data.lane_store`: pieces = paths up to 100 m with lane / connector segments, 0.5 m polylines).
Origin: the owner's questions of 2026-11-23 (every section a Catmull-Rom spline through the nodes; should the nodes pull information from their neighbours; support points at intersections?). Numbers: memory note
lane-catmull-rom-check-2026-11-23.

    wsl bash tools/env/_py14.sh curve_repr_check.py [repr|noise|kinks|boundaries|splits|curvature|nodeerror|paircorr|smooth|all] [--every 12]      # VAL frames, every 12th (73 frames, 1090 paths)
    wsl bash tools/env/_py14.sh curve_repr_check.py --selftest

  repr        interior nodes per 100 m that a polyline / uniform Catmull-Rom / centripetal Catmull-Rom / cubic Hermite with the TRUE heading (the present connector with a perfect heading, handles 1/3 of the chord) needs to stay
              within a symmetric Hausdorff tolerance (greedy insertion of the worst point, as Douglas-Peucker does); distances are point-to-segment on the sampled curve (24 samples per segment)
  noise       error at a fixed 7 m node spacing for node noise sigma 0 / 0.2 / 0.5 m (independent per node)
  kinks       turning angle per 1 m of the GT (sharp corners where a spline overshoots)
  boundaries  Catmull-Rom through nodes at 7 m spacing: with and without a node forced at every lane <-> connector boundary (the intersection entry / exit): error near the boundaries and node density
  splits      angles at the split / merge events of the GT between the path directions on each side (can ONE shared node carry ONE tangent?)
  curvature   the same Catmull-Rom error binned by the local turning rate of the GT (7 m and 3.5 m node spacing): where would extra nodes pay
  nodeerror   the position errors of the B' tokens of VAL against their GT nodes: lateral / longitudinal rms and the correlation of the lateral error between nodes 1 .. 5 hops (about 2 m each) apart
  paircorr    the correlation of the 2-D position error between two nodes at distance d, on the SAME lane (<= 10 hops of the GT graph) or on another lane: along the lane or across lanes?
  smooth      nodes every 7 m with independent noise: the spline through the raw nodes against a spline through nodes that were smoothed with their neighbours (3 / 5 nodes, quadratic fit): how much do neighbours help
The curves (`cr_uniform`, `cr_centripetal`, `hermite_true`) are the building blocks of a spline read-out; a read-out of the champion's nodes would import them from here."""
import os
import sys

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for sub_ in ("training/data", "training/data/prep", "net/prep", "net/model", "net/loss", "training/train", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import numpy as np  # noqa: E402

NS = 24                                                                         # samples per spline segment
SPACING = 7.0                                                                   # m, the fixed node spacing of the noise / boundary / smoothing checks


def cr_uniform(P, ns=NS):
    """uniform Catmull-Rom through the nodes P (n, 2): the ends get a reflected ghost node; (m, 2) curve samples, the last one is P[-1]"""
    P = np.asarray(P, float)
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    t = np.linspace(0, 1, ns, endpoint=False)[:, None]
    out = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-1][None])
    return np.vstack(out)


def cr_centripetal(P, ns=NS, alpha=0.5):
    """centripetal Catmull-Rom (Barry-Goldman); alpha 0.5 avoids cusps and loops at uneven node spacing"""
    P = np.asarray(P, float)
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** alpha
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** alpha
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** alpha
        t = np.linspace(t1, t2, ns, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(P[-1][None])
    return np.vstack(out)


def hermite_true(P, T, ns=NS):
    """cubic Bezier / Hermite per segment with the given unit tangents T at the nodes, handles 1/3 of the chord (the present connector with a perfect heading)"""
    t = np.linspace(0, 1, ns, endpoint=False)[:, None]
    out = []
    for i in range(len(P) - 1):
        L = np.linalg.norm(P[i + 1] - P[i])
        b0, b3 = P[i], P[i + 1]
        b1, b2 = b0 + T[i] * L / 3, b3 - T[i + 1] * L / 3
        out.append((1 - t) ** 3 * b0 + 3 * (1 - t) ** 2 * t * b1 + 3 * (1 - t) * t ** 2 * b2 + t ** 3 * b3)
    out.append(P[-1][None])
    return np.vstack(out)


def poly(P, ns=NS):
    t = np.linspace(0, 1, ns, endpoint=False)[:, None]
    out = [P[i] * (1 - t) + P[i + 1] * t for i in range(len(P) - 1)]
    out.append(P[-1][None])
    return np.vstack(out)


def seg_dist(Q, C):
    """distance of every point of Q (n, 2) to the polyline C (m, 2): point to SEGMENT, not to the nearest vertex (vertex distances make a sampled curve look 0.1 - 0.4 m off)"""
    a, b = C[:-1], C[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    t = np.clip(((Q[:, None] - a[None]) * ab[None]).sum(2) / L2[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(Q[:, None] - proj, axis=2).min(1)


def hausdorff(A, B):
    return max(seg_dist(A, B).max(), seg_dist(B, A).max())


def tangents(G):
    T = np.gradient(G, axis=0)
    return T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)


def recon(method, G, TG, idx):
    P = G[idx]
    if method == "polyline":
        return poly(P)
    if method == "cr_uniform":
        return cr_uniform(P)
    if method == "cr_centripetal":
        return cr_centripetal(P)
    return hermite_true(P, TG[idx])


def greedy(method, G, TG, eps, maxn=60):
    """the node indices (ends included) that keep the rebuilt curve within eps of the GT polyline G (both directions)"""
    idx = {0, len(G) - 1}
    while len(idx) < maxn:
        C = recon(method, G, TG, np.array(sorted(idx)))
        d_gc, d_cg = seg_dist(G, C), seg_dist(C, G)
        if d_gc.max() <= eps and d_cg.max() <= eps:
            break
        k = int(np.argmax(d_gc)) if d_gc.max() >= d_cg.max() else int(np.argmin(np.linalg.norm(G - C[int(np.argmax(d_cg))], axis=1)))   # the worst GT point, or the GT point under the worst bulge
        if k in idx:
            break
        idx.add(k)
    return sorted(idx)


def load(every):
    import lane_gt_data as LD
    import lane_sdgraph_data as SG
    lg, _ = LD.lane_store()
    tok = SG.load("val")["tokens"][::every]
    return [lg.load(t) for t in tok]


def paths_of(frames):
    """GT paths (dense polylines) and sections (the lane / connector segments of a path)"""
    paths, sections = [], []
    for g in frames:
        for p in g["pieces"]:
            G = np.asarray(p["pts"], float)
            paths.append(G)
            for s in p["segments"]:
                S = G[s["i0"]: s["i1"] + 1]
                if len(S) >= 12:
                    sections.append(S)
    return paths, sections


def arclen(G):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(G, axis=0), axis=1))]


def even_nodes(G, step=SPACING):
    s = arclen(G)
    k = np.unique(np.r_[np.round(np.arange(0, s[-1], step) / 0.5).astype(int), len(G) - 1])
    return k[k < len(G)]


def report_repr(paths, sections):
    methods = ["polyline", "cr_uniform", "cr_centripetal", "hermite_true_heading"]
    for name, geo in (("PATHS (the GT pieces, up to 100 m)", paths), ("SECTIONS (lane / connector segments)", sections)):
        total = sum(arclen(G)[-1] for G in geo)
        print(f"\n{name}: {total / 1000:.1f} km; interior nodes per 100 m (the two chain ends are not counted) to stay within eps (symmetric Hausdorff)")
        print("eps m    " + "".join(f"{m:>24}" for m in methods))
        for eps in (0.2, 0.3, 0.5, 1.0):
            row = [sum(len(greedy(m, G, tangents(G), eps)) - 2 for G in geo) / total * 100 for m in methods]
            print(f"{eps:5.1f}    " + "".join(f"{v:24.2f}" for v in row))


def report_noise(paths):
    print(f"\nnoise: nodes every {SPACING:.0f} m, independent Gaussian node noise sigma; distance GT -> curve, rms / 95 % (m)")
    rng = np.random.default_rng(0)
    ms = (("polyline", poly), ("cr_uniform", cr_uniform), ("cr_centripetal", cr_centripetal))
    print("sigma    " + "".join(f"{m:>24}" for m, _ in ms))
    for sg in (0.0, 0.2, 0.5):
        res = {m: [] for m, _ in ms}
        for G in paths[:400]:
            if arclen(G)[-1] < 3 * SPACING:
                continue
            k = even_nodes(G)
            P = G[k] + rng.normal(0, sg, (len(k), 2))
            for m, f in ms:
                res[m].append(seg_dist(G, f(P)))
        print(f"{sg:5.1f}    " + "".join(f"{np.sqrt(np.mean(np.concatenate(res[m]) ** 2)):10.3f} / {np.percentile(np.concatenate(res[m]), 95):9.3f}" for m, _ in ms))


def report_kinks(paths):
    a = np.concatenate([np.degrees(np.arccos(np.clip((tangents(G)[::2][1:] * tangents(G)[::2][:-1]).sum(1), -1, 1))) for G in paths])
    print("\nturning angle per 1 m of the GT paths: " + ", ".join(f"> {th} deg: {100 * (a > th).mean():.2f} %" for th in (5, 10, 20)) + f"; max {a.max():.1f} deg over {len(a)} steps")


def report_boundaries(frames):
    """Catmull-Rom through nodes at 7 m spacing: (A) evenly along the path, ignoring the segment kinds; (B) additionally a node at every lane <-> connector boundary (and the even nodes thinned so that none is closer than 2 m to it)"""
    near = {"A": [], "B": [], "A_far": [], "B_far": []}
    cnt = {"A": 0, "B": 0}
    length = 0.0
    n_b = 0
    for g in frames:
        for p in g["pieces"]:
            G = np.asarray(p["pts"], float)
            segs = p["segments"]
            bi = [s["i0"] for a, s in zip(segs[:-1], segs[1:]) if a["kind"] != s["kind"] for s in [s]]
            if not bi or arclen(G)[-1] < 3 * SPACING:
                continue
            s_ = arclen(G)
            kA = even_nodes(G)
            kB = np.array(sorted(set(kA[np.abs(s_[kA][:, None] - s_[bi][None]).min(1) >= 2.0].tolist()) | set(bi) | {0, len(G) - 1}))
            dB = np.abs(s_[:, None] - s_[bi][None]).min(1)
            for tag, k in (("A", kA), ("B", kB)):
                d = seg_dist(G, cr_uniform(G[k]))
                near[tag].append(d[dB <= 4.0])
                near[tag + "_far"].append(d[dB > 4.0])
                cnt[tag] += len(k) - 2
            length += s_[-1]
            n_b += len(bi)
    print(f"\nboundaries: {n_b} lane <-> connector boundaries on {length / 1000:.1f} km of paths (paths that have one); Catmull-Rom through nodes every {SPACING:.0f} m; distance GT -> curve, rms / 95 % / max (m)")
    print("                                   within 4 m of a boundary          elsewhere            interior nodes per 100 m")
    for tag, name in (("A", "A: even nodes only"), ("B", "B: + a node at each boundary")):
        n, f = np.concatenate(near[tag]), np.concatenate(near[tag + "_far"])
        print(f"{name:34s} {np.sqrt(np.mean(n ** 2)):7.3f} / {np.percentile(n, 95):6.3f} / {n.max():6.3f}   {np.sqrt(np.mean(f ** 2)):7.3f} / {np.percentile(f, 95):6.3f} / {f.max():6.3f}   {cnt[tag] / length * 100:8.2f}")


def unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def report_splits(frames):
    """the split / merge EVENTS of the GT (`g['events']`: the position where paths separate or join, `dirs_in` / `dirs_out` = the unit directions of the paths on each side)"""
    s_in_out, s_out_out, m_in_in, m_in_out = [], [], [], []
    for g in frames:
        for e in g["events"]:
            if e["type"] not in ("split", "merge"):
                continue
            di, do = unit(e["dirs_in"]), unit(e["dirs_out"])
            if e["type"] == "split" and len(di) >= 1 and len(do) >= 2:
                s_in_out += [angle(di[0], d) for d in do]
                s_out_out.append(angle(do[0], do[1]))
            if e["type"] == "merge" and len(di) >= 2 and len(do) >= 1:
                m_in_in.append(angle(di[0], di[1]))
                m_in_out += [angle(d, do[0]) for d in di]

    def row(name, v):
        if not len(v):
            return f"{name:46s} n     0"
        v = np.array(v)
        return f"{name:46s} n {len(v):4d}  median {np.median(v):5.1f}  p90 {np.percentile(v, 90):5.1f}  max {v.max():5.1f} deg   > 5 deg: {100 * (v > 5).mean():5.1f} %  > 15 deg: {100 * (v > 15).mean():5.1f} %"
    print("\nsplit / merge events of the GT: angle between the path directions at the event")
    print(row("split: incoming path vs each outgoing path", s_in_out))
    print(row("split: the two outgoing paths against each other", s_out_out))
    print(row("merge: each incoming path vs the outgoing path", m_in_out))
    print(row("merge: the two incoming paths against each other", m_in_in))


def report_curvature(paths):
    """Catmull-Rom through exact nodes at 7 m and at 3.5 m spacing: the distance GT -> curve binned by the local turning rate of the GT (rad per m; 0.05 = a radius of 20 m): where do the errors sit, where would extra nodes pay"""
    bins = [(0.0, 0.01), (0.01, 0.03), (0.03, 0.06), (0.06, 1.0)]
    print("\ncurvature: Catmull-Rom through exact nodes; distance GT -> curve (rms / max, m) by the local turning rate of the GT (rad per m)")
    print("spacing    " + "".join(f"{f'{lo:.2f} - {hi:.2f}':>20}" for lo, hi in bins) + "      share of the length per bin")
    for step in (7.0, 3.5):
        d_all, k_all = [], []
        for G in paths:
            if arclen(G)[-1] < 3 * step:
                continue
            T = tangents(G)
            turn = np.r_[0, np.degrees(np.arccos(np.clip((T[1:] * T[:-1]).sum(1), -1, 1)))] * np.pi / 180 / 0.5
            turn = np.convolve(turn, np.ones(5) / 5, mode="same")
            k = even_nodes(G, step)
            d_all.append(seg_dist(G, cr_uniform(G[k])))
            k_all.append(turn)
        d, tr = np.concatenate(d_all), np.concatenate(k_all)
        cells = []
        for lo, hi in bins:
            m = (tr >= lo) & (tr < hi)
            cells.append(f"{np.sqrt(np.mean(d[m] ** 2)):8.3f} / {d[m].max():6.3f}" if m.any() else " " * 20)
        print(f"{step:5.1f} m   " + "".join(f"{c:>20}" for c in cells) + "      " + " ".join(f"{100 * ((tr >= lo) & (tr < hi)).mean():5.1f} %" for lo, hi in bins))


def smooth_nodes(P, k):
    """each node replaced by the value at its position of a quadratic least-squares fit over its k nearest neighbours in the chain (k = 3 or 5; the ends stay)"""
    Q = P.copy()
    h = k // 2
    for i in range(h, len(P) - h):
        x = np.arange(-h, h + 1)
        A = np.vander(x, 3)
        coef = np.linalg.lstsq(A, P[i - h: i + h + 1], rcond=None)[0]
        Q[i] = coef[-1]
    return Q


def report_smooth(paths):
    print(f"\nsmoothing: nodes every {SPACING:.0f} m with INDEPENDENT noise sigma; Catmull-Rom through the raw nodes vs through nodes smoothed with their neighbours (quadratic fit over 3 / 5 nodes); distance GT -> curve, rms (m)")
    rng = np.random.default_rng(1)
    print("sigma    raw nodes     3-node fit    5-node fit")
    for sg in (0.2, 0.5, 1.0):
        res = {"raw": [], "k3": [], "k5": []}
        for G in paths[:400]:
            if arclen(G)[-1] < 5 * SPACING:
                continue
            k = even_nodes(G)
            P = G[k] + rng.normal(0, sg, (len(k), 2))
            for name, Q in (("raw", P), ("k3", smooth_nodes(P, 3)), ("k5", smooth_nodes(P, 5))):
                res[name].append(seg_dist(G, cr_uniform(Q)))
        print(f"{sg:5.1f}    " + "".join(f"{np.sqrt(np.mean(np.concatenate(res[m]) ** 2)):10.3f}    " for m in ("raw", "k3", "k5")))


def report_node_error(every=4):
    """the position errors of B' tokens (VAL, cache seed 0) against the GT node they are assigned to (`gt_id`): lateral / longitudinal rms and the correlation of the lateral error between nodes h hops (about 2 m) apart along the GT graph (shared or independent errors?)"""
    import lane_ptok_data as LP
    import lane_tok_data as LT
    ds = LP.PTokData("val", 0, "e292_a", None, 0.0)
    td = LT.TokData("val", res=2.0, feats=False)
    lat_pairs = {h: [] for h in range(1, 6)}
    lon_pairs = {h: [] for h in range(1, 6)}
    lats, lons, rngs, n_tok = [], [], [], 0
    for i in range(0, len(ds), every):
        it = ds[i]
        t = td.table(i)
        gid = np.asarray(it["label"]["gt_id"])
        valid = np.asarray(it["valid"], bool)
        pos = np.asarray(it["pos"], float)
        m = valid & (gid >= 0) & (gid < len(t["pos"]))
        if not m.any():
            continue
        e = {}
        for k in np.flatnonzero(m):
            e.setdefault(int(gid[k]), []).append(pos[k])
        err_lat, err_lon = {}, {}
        for g, ps in e.items():
            d = np.mean(ps, axis=0) - np.asarray(t["pos"][g], float)
            u = np.asarray(t["dirv"][g], float)
            u = u / max(np.linalg.norm(u), 1e-9)
            err_lon[g] = float(d @ u)
            err_lat[g] = float(d @ np.array([-u[1], u[0]]))
        lats += list(err_lat.values())
        rngs += [float(np.linalg.norm(np.asarray(t["pos"][g], float))) for g in err_lat]
        lons += list(err_lon.values())
        n_tok += int(m.sum())
        succ = {}
        for a, b in zip(t["src"], t["dst"]):
            succ.setdefault(int(a), []).append(int(b))
        for g in err_lat:
            front = [g]
            for h in range(1, 6):
                front = [b for a in front for b in succ.get(a, [])]
                for b in front:
                    if b in err_lat:
                        lat_pairs[h].append((err_lat[g], err_lat[b]))
                        lon_pairs[h].append((err_lon[g], err_lon[b]))
    lats, lons, rngs = np.array(lats), np.array(lons), np.array(rngs)
    print(f"VAL cache seed 0, every {every}th frame: {len(lats)} GT nodes with an assigned token ({n_tok} tokens)")
    print(f"lateral error rms {np.sqrt(np.mean(lats ** 2)):.3f} m (median |e| {np.median(np.abs(lats)):.3f}, p90 {np.percentile(np.abs(lats), 90):.3f}); longitudinal rms {np.sqrt(np.mean(lons ** 2)):.3f} m")
    print("lateral error rms by range from the ego: " + ", ".join(f"{lo} - {hi} m: {np.sqrt(np.mean(lats[(rngs >= lo) & (rngs < hi)] ** 2)):.3f} (n {int(((rngs >= lo) & (rngs < hi)).sum())})" for lo, hi in ((0, 15), (15, 30), (30, 50), (50, 100))))
    print("correlation of the error between GT nodes h hops (about 2 m each) apart:   h:  " + "  ".join(f"{h}" for h in range(1, 6)))
    for name, pairs in (("lateral     ", lat_pairs), ("longitudinal", lon_pairs)):
        print(f"  {name}" + " " * 52 + "  ".join(f"{np.corrcoef(np.array(pairs[h]).T)[0, 1]:.2f}" for h in range(1, 6)) + "   n " + str([len(pairs[h]) for h in range(1, 6)]))


def report_pair_corr(every=4):
    """does a node's position error resemble that of nodes on the SAME lane (connected within 10 hops of the GT graph, about 20 m) or that of nodes on OTHER lanes at the same Euclidean distance? Global 2-D error vectors of the B' tokens against their GT
    nodes (VAL, cache seed 0); correlation = sum(e_a . e_b) / sqrt(sum |e_a|^2 sum |e_b|^2) per distance bin"""
    import lane_ptok_data as LP
    import lane_tok_data as LT
    ds = LP.PTokData("val", 0, "e292_a", None, 0.0)
    td = LT.TokData("val", res=2.0, feats=False)
    edges = [0, 2, 4, 6, 10, 15, 25]
    acc = {c: np.zeros((len(edges) - 1, 3)) for c in ("same", "other")}
    cnt = {c: np.zeros(len(edges) - 1, int) for c in ("same", "other")}
    for i in range(0, len(ds), every):
        it = ds[i]
        t = td.table(i)
        gid = np.asarray(it["label"]["gt_id"])
        valid = np.asarray(it["valid"], bool)
        pos = np.asarray(it["pos"], float)
        m = valid & (gid >= 0) & (gid < len(t["pos"]))
        e = {}
        for k in np.flatnonzero(m):
            e.setdefault(int(gid[k]), []).append(pos[k])
        ids = np.array(sorted(e))
        if len(ids) < 2:
            continue
        gp = np.asarray(t["pos"], float)[ids]
        ev = np.array([np.mean(e[g], axis=0) for g in ids]) - gp
        adj = {}
        for a_, b_ in zip(t["src"], t["dst"]):
            adj.setdefault(int(a_), set()).add(int(b_))
            adj.setdefault(int(b_), set()).add(int(a_))
        index = {int(g): n for n, g in enumerate(ids)}
        near = np.zeros((len(ids), len(ids)), bool)
        for g in ids:
            seen, front = {int(g)}, {int(g)}
            for _ in range(10):
                front = {y for x in front for y in adj.get(x, ())} - seen
                seen |= front
            for x in seen:
                if x in index:
                    near[index[int(g)], index[x]] = True
        D = np.linalg.norm(gp[:, None] - gp[None], axis=2)
        iu = np.triu_indices(len(ids), 1)
        d, same = D[iu], near[iu]
        dot = (ev[:, None] * ev[None]).sum(2)[iu]
        n2 = (ev ** 2).sum(1)
        na, nb = n2[iu[0]], n2[iu[1]]
        for b in range(len(edges) - 1):
            sel = (d >= edges[b]) & (d < edges[b + 1])
            for c, mm in (("same", sel & same), ("other", sel & ~same)):
                acc[c][b] += [dot[mm].sum(), na[mm].sum(), nb[mm].sum()]
                cnt[c][b] += int(mm.sum())
    print("correlation of the 2-D position error (global frame) between two nodes at Euclidean distance d, on the same lane (<= 10 hops of the GT graph) or on another lane")
    print("d (m)        " + "".join(f"{f'{edges[b]} - {edges[b + 1]}':>14}" for b in range(len(edges) - 1)))
    for c, name in (("same", "same lane"), ("other", "other lane")):
        r = [acc[c][b][0] / np.sqrt(max(acc[c][b][1] * acc[c][b][2], 1e-12)) for b in range(len(edges) - 1)]
        print(f"{name:12s} " + "".join(f"{v:14.2f}" for v in r) + "   pairs " + str(list(cnt[c])))


def selftest():
    x = np.arange(0, 30.1, 7.5)
    L = np.stack([x, np.zeros_like(x)], 1)
    C = cr_uniform(L)
    assert np.abs(C[:, 1]).max() < 1e-9 and np.all(np.diff(C[:, 0]) >= -1e-9), "a spline through collinear nodes is the line"
    assert np.allclose(C[::NS][: len(L)], L), "the spline passes through its nodes"
    assert abs(seg_dist(np.array([[1.0, 1.0]]), np.array([[0.0, 0.0], [4.0, 0.0]]))[0] - 1.0) < 1e-12, "point to segment, not to the nearest vertex"
    R, ang = 50.0, np.linspace(0, 0.9, 400)
    G = np.stack([R * np.sin(ang), R * (1 - np.cos(ang))], 1)
    TG = tangents(G)
    k = np.array([0, 100, 200, 300, 399])
    e = {m: hausdorff(G, recon(m, G, TG, k)) for m in ("polyline", "cr_uniform", "cr_centripetal", "hermite")}
    sag = R * (1 - np.cos(0.5 * 0.9 / 4 * 1.0))
    assert e["polyline"] > 0.9 * sag * 0.9 and e["cr_uniform"] < e["polyline"] and e["hermite"] < 0.02 and abs(e["cr_uniform"] - e["cr_centripetal"]) < 0.05, e
    assert greedy("polyline", L, tangents(L), 0.1) == [0, len(L) - 1] and len(greedy("polyline", G, TG, 0.05)) > 2
    P = np.stack([np.arange(7.0) * 7.0, np.zeros(7)], 1)
    P[3, 1] = 2.0
    assert abs(smooth_nodes(P, 5)[3, 1]) < abs(P[3, 1]) and np.allclose(smooth_nodes(P, 3)[0], P[0]), "a smoother pulls an outlier node toward its neighbours and leaves the ends"
    print("curve_repr_check selftest ok")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    if "--selftest" in sys.argv:
        return selftest()
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 12
    what = [a for a in sys.argv[1:] if a in ("repr", "noise", "kinks", "boundaries", "splits", "curvature", "nodeerror", "paircorr", "smooth", "all")] or ["all"]
    frames = load(every)
    paths, sections = paths_of(frames)
    print(f"{len(frames)} frames, {len(paths)} paths, {len(sections)} sections (>= 6 m)")
    for w, f in (("repr", lambda: report_repr(paths, sections)), ("noise", lambda: report_noise(paths)), ("kinks", lambda: report_kinks(paths)), ("boundaries", lambda: report_boundaries(frames)),
                 ("splits", lambda: report_splits(frames)), ("curvature", lambda: report_curvature(paths)), ("nodeerror", lambda: report_node_error()), ("paircorr", lambda: report_pair_corr()), ("smooth", lambda: report_smooth(paths))):
        if "all" in what or w in what:
            f()
            sys.stdout.flush()


if __name__ == "__main__":
    main()
