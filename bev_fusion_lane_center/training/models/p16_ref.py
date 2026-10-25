"""Reference implementation (numpy, no learning) of the verified-ray hypothesis tree with an ORACLE verification (distance to toy lanes).
It defines the geometry, the survival algebra, the hypothesis selection, the fixed-shape tree, the merge into a DAG and the time update that the network version must reproduce; test_p16_spec.py runs the spec tests against it.
Conventions: metres in the ego frame (x forward, y left), angles in radians internally, degrees in the configuration; phi = bearing of a ray relative to the tip heading."""
import math

import numpy as np
from scipy.spatial import cKDTree

CFG = dict(
    phis=(0, 9, -9, 20, -20, 50, -50, 90, -90, 125, -125, 180),  # ray bearings (deg)
    r_cap=24.0, r_floor=4.0, phi_c=20.0,  # envelope r(phi) = max(r_floor, r_cap / (1 + (phi / phi_c)^2))
    rings=(4.0, 8.0, 12.0, 18.0, 24.0),  # shared sample rings (m)
    sigma=0.6,  # oracle verification width (m)
    v_floor=0.02, v_ceil=0.99,
    prior_deg=45.0, prior_floor=0.02,  # direction prior of a continuation: exp(-(phi / prior_deg)^2)
    cov_r=2.5,  # an endpoint closer than this to a committed point is masked
    nms_seg=1.2, nms_r=2.0,  # an endpoint closer than nms_seg to the segment of a longer, comparably valid candidate is redundant; endpoints closer than nms_r are duplicates
    alive=0.3,  # validity threshold of a living hypothesis
    skip_penalty=0.5,  # tolerant survival: one missing sample costs this factor
    merge_r=1.5, merge_deg=30.0,
    snap_r=1.0, snap_end=1.5,  # an edge that passes closer than snap_r to a vertex (not near its ends) is split there
)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def wrap_deg(a):
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def envelope(phi_deg, cfg=CFG):
    phi = np.asarray(phi_deg, float)
    return np.maximum(cfg["r_floor"], cfg["r_cap"] / (1.0 + (phi / cfg["phi_c"]) ** 2))


# ---------------------------------------------------------------- M1 pattern
def make_rays(c, th, cfg=CFG, bipolar=False):
    """rays at tip c (2,) with heading th; bipolar = seed without heading (two lobes)"""
    if bipolar:
        base = (0, 9, -9, 20, -20, 50, -50, 90)
        phis = sorted({wrap_deg(a) for a in base} | {wrap_deg(a + 180) for a in base})
    else:
        phis = [wrap_deg(a) for a in cfg["phis"]]
    rays = []
    for a in phis:
        a_ax = min(abs(wrap_deg(a)), abs(wrap_deg(a - 180))) if bipolar else abs(wrap_deg(a))
        r = float(envelope(a_ax, cfg))
        radii = [rk for rk in cfg["rings"] if rk < r - 1e-6] + [r]
        d = np.array([math.cos(th + math.radians(a)), math.sin(th + math.radians(a))])
        pts = np.array([np.asarray(c, float) + rr * d for rr in radii])
        rays.append(dict(phi=a, phi_axis=a_ax, r=r, radii=np.array(radii), pts=pts))
    return rays


# ---------------------------------------------------------------- M2 verification and survival
class Scene:
    """toy lanes (list of polylines); the oracle verification is a function of the distance to the nearest lane point"""

    def __init__(self, lanes, ds=0.25):
        pts = []
        for p in lanes:
            p = np.asarray(p, float)
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            q = np.linspace(0, s[-1], max(int(s[-1] / ds), 2))
            pts.append(np.c_[np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])])
        self.lanes = [np.asarray(p, float) for p in lanes]
        self.pts = np.concatenate(pts)
        self.tree = cKDTree(self.pts)

    def dist(self, p):
        return self.tree.query(np.atleast_2d(p))[0]

    def verify(self, pts, cfg=CFG):
        d = self.dist(pts)
        return np.clip(np.exp(-((d / cfg["sigma"]) ** 2)), cfg["v_floor"], cfg["v_ceil"])


def survival(v):
    """T_k = v_1 ... v_k; P[0] = 1 - v_1 (no sample verified), P[k] = T_k (1 - v_(k+1)) for k < K, P[K] = T_K. sum(P) = 1."""
    v = np.asarray(v, float)
    T = np.cumprod(v)
    nxt = np.r_[v[1:], 0.0]
    return T, np.r_[1.0 - v[0], T * (1.0 - nxt)]


def survival_tolerant(v, cfg=CFG):
    """strict survival with one allowed missing sample at a penalty (occlusion): T_k = max(T_(k-1) v_k, skip * T_(k-2) v_k)"""
    v = np.asarray(v, float)
    T = np.zeros(len(v))
    T[0] = v[0]
    for k in range(1, len(v)):
        a = T[k - 1] * v[k]
        b = (T[k - 2] if k >= 2 else 1.0) * cfg["skip_penalty"] * v[k]
        T[k] = max(a, b)
    return T


def ray_candidate(scene, ray, cfg=CFG, tolerant=False, use_prior=True):
    v = scene.verify(ray["pts"], cfg)
    if tolerant:
        T = survival_tolerant(v, cfg)
        ok = np.where(T >= cfg["alive"])[0]
        k = int(ok.max()) if len(ok) else 0
    else:
        T, P = survival(v)
        k = int(np.argmax(P[1:]))  # most probable last verified sample (0-based)
    prior = max(cfg["prior_floor"], math.exp(-((ray["phi_axis"] / cfg["prior_deg"]) ** 2))) if use_prior else 1.0
    return dict(ray=ray, v=v, T=T, k=k, end=ray["pts"][k], nu=float(T[k]) * prior, prior=prior)


# ---------------------------------------------------------------- M3 hypothesis selection
def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def select(cands, tip, b, cfg=CFG):
    """masks are applied before; redundancy = dominated by a longer comparably valid candidate whose segment passes the endpoint, or a duplicate endpoint; returns the b best (padded with dummies)"""
    tip = np.asarray(tip, float)
    reach = [float(np.linalg.norm(c["end"] - tip)) for c in cands]
    red = set()
    for i, ci in enumerate(cands):
        for j, cj in enumerate(cands):
            if i == j:
                continue
            if reach[j] > reach[i] + 1e-6 and cj["nu"] >= 0.5 * ci["nu"] and seg_dist(ci["end"], tip, cj["end"]) < cfg["nms_seg"]:
                red.add(i)
                break
    order = sorted([i for i in range(len(cands)) if i not in red], key=lambda i: (-cands[i]["nu"], i))
    kept = []
    for i in order:
        if all(np.linalg.norm(cands[i]["end"] - cands[j]["end"]) > cfg["nms_r"] for j in kept):
            kept.append(i)
    return [cands[i] for i in kept[:b]], [cands[i] for i in range(len(cands)) if i not in kept[:b]]


def expand(scene, pos, th, committed, b=2, cfg=CFG, tolerant=False, bipolar=False):
    """children of one tip: dicts pos, th, nu, alive, dummy, phi; ordered left to right (bearing); always exactly b entries (fixed shape)"""
    pos = np.asarray(pos, float)
    rays = make_rays(pos, th, cfg, bipolar)
    cands = [ray_candidate(scene, r, cfg, tolerant, use_prior=not bipolar) for r in rays]
    if len(committed):
        ct = cKDTree(np.asarray(committed, float))
        for c in cands:
            if ct.query(c["end"])[0] < cfg["cov_r"]:
                c["nu"] *= 0.02
    kept, _ = select(cands, pos, b, cfg)
    out = []
    for c in kept:
        e = c["end"]
        out.append(dict(pos=e.copy(), th=math.atan2(e[1] - pos[1], e[0] - pos[0]), nu=c["nu"], alive=c["nu"] >= cfg["alive"], dummy=False, phi=c["ray"]["phi"]))
    out.sort(key=lambda o: -wrap(o["th"] - th))  # left (positive bearing) first
    while len(out) < b:
        out.append(dict(pos=pos.copy(), th=th, nu=0.0, alive=False, dummy=True, phi=0.0))
    return out, cands


# ---------------------------------------------------------------- M4 fixed-shape tree
def grow_tree(scene, root_pos, root_th, D=4, b=2, committed0=(), cfg=CFG, tolerant=False, bipolar_root=False):
    """node i of level d has parent index parent[i]; level d has b^d nodes per root (dead parents give dummy children): shape is fixed. pi = min over the path of the validity (bottleneck)."""
    nodes = [dict(level=0, parent=-1, pos=np.asarray(root_pos, float), th=float(root_th), nu=1.0, pi=1.0, alive=True, dummy=False)]
    frontier = [0]
    for d in range(1, D + 1):
        new = []
        for ni in frontier:
            n = nodes[ni]
            if n["alive"] and not n["dummy"]:
                anc, k = [], ni
                while k >= 0:
                    anc.append(nodes[k]["pos"])
                    k = nodes[k]["parent"]
                ch, _ = expand(scene, n["pos"], n["th"], list(committed0) + anc[1:], b, cfg, tolerant, bipolar=(bipolar_root and ni == 0))
            else:
                ch = [dict(pos=n["pos"].copy(), th=n["th"], nu=0.0, alive=False, dummy=True, phi=0.0) for _ in range(b)]
            for c in ch:
                pi = min(n["pi"], c["nu"])
                nodes.append(dict(level=d, parent=ni, pos=c["pos"], th=c["th"], nu=c["nu"], pi=pi, alive=(pi >= cfg["alive"] and not c["dummy"]), dummy=c["dummy"]))
                new.append(len(nodes) - 1)
        frontier = new
    return nodes


def level_counts(nodes):
    out = {}
    for n in nodes:
        out[n["level"]] = out.get(n["level"], 0) + 1
    return [out[k] for k in sorted(out)]


def prune_top_m(nodes, M):
    """beam budget: keep the M alive nodes with the highest pi per level (ties by index); returns a set of indices (selection only)"""
    keep = set()
    for d in sorted({n["level"] for n in nodes}):
        idx = [i for i, n in enumerate(nodes) if n["level"] == d and n["alive"]]
        idx.sort(key=lambda i: (-nodes[i]["pi"], i))
        keep |= set(idx[:M])
    return keep


def merge_nodes(nodes, cfg=CFG):
    """tree -> DAG: alive nodes closer than merge_r with a similar heading are one vertex (position = pi-weighted mean); edges parent -> child between clusters"""
    alive = [i for i, n in enumerate(nodes) if n["alive"]]
    order = sorted(alive, key=lambda i: (-nodes[i]["pi"], i))
    cl = {}
    centres = []
    for i in order:
        n = nodes[i]
        hit = None
        for ci, cidx in enumerate(centres):
            m = nodes[cidx]
            if np.linalg.norm(n["pos"] - m["pos"]) < cfg["merge_r"] and abs(math.degrees(wrap(n["th"] - m["th"]))) < cfg["merge_deg"]:
                hit = ci
                break
        if hit is None:
            centres.append(i)
            hit = len(centres) - 1
        cl[i] = hit
    pos = np.zeros((len(centres), 2))
    w = np.zeros(len(centres))
    for i, c in cl.items():
        pos[c] += nodes[i]["pi"] * nodes[i]["pos"]
        w[c] += nodes[i]["pi"]
    pos /= np.maximum(w[:, None], 1e-9)
    edges = set()
    for i, c in cl.items():
        p = nodes[i]["parent"]
        if p >= 0 and p in cl and cl[p] != c:
            edges.add((cl[p], c))
    return pos, split_edges_at_vertices(pos, sorted(edges), cfg)


def split_edges_at_vertices(pos, edges, cfg=CFG, max_iter=200):
    """M4b: where a hypothesis path joins an existing lane between two vertices, the long edge is split at the vertex that lies on it; this is what turns two overlapping paths into a merge (in-degree 2)"""
    E = set(edges)
    for _ in range(max_iter):
        changed = False
        for (a, b) in sorted(E):
            pa, pb = pos[a], pos[b]
            ab = pb - pa
            for v in range(len(pos)):
                if v in (a, b):
                    continue
                t = float(np.dot(pos[v] - pa, ab) / max(np.dot(ab, ab), 1e-12))
                if 0.0 < t < 1.0 and seg_dist(pos[v], pa, pb) < cfg["snap_r"] and np.linalg.norm(pos[v] - pa) > cfg["snap_end"] and np.linalg.norm(pos[v] - pb) > cfg["snap_end"]:
                    E.discard((a, b))
                    E.add((a, v))
                    E.add((v, b))
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    return sorted(E)


# ---------------------------------------------------------------- M6 time
def se2_apply(xy, th, dx, dy, dpsi):
    """express points of frame t-1 in frame t for an ego motion (dx, dy, dpsi) given in frame t-1: p_t = R(-dpsi) (p - d)"""
    c, s = math.cos(-dpsi), math.sin(-dpsi)
    q = np.asarray(xy, float) - np.array([dx, dy])
    return np.c_[c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1]], np.asarray(th, float) - dpsi


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fuse(pi_prev, nu_now, a=0.9, b=1.0):
    """log-odds fusion of the warped previous validity and the current evidence"""
    return 1.0 / (1.0 + np.exp(-(a * logit(pi_prev) + b * logit(nu_now))))
