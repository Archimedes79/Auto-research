"""Loop 16: inference of the builders - tree -> DAG assembly (merge of duplicates, edge split at vertices on edges, pieces) and the evaluation sweep over the validity threshold.
Usage: imported by p16_train.py / p16_diag.py.  Training-free numpy code (inference only: thresholds and clusters are allowed here)."""
import bisect
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_loss as PL  # noqa: E402

TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class Union:
    """incremental union of the paths of the tree: vertices, directed edges; a new node that lies on an existing edge is snapped onto it (the edge is split there: merge / fork), duplicates are not added"""

    def __init__(self):
        self.V, self.E, self.adj = [], set(), {}

    def add_vertex(self, p):
        self.V.append(np.asarray(p, float))
        return len(self.V) - 1

    def nearest_vertex(self, p):
        if not self.V:
            return -1, 1e9
        d = np.linalg.norm(np.asarray(self.V) - p, axis=1)
        k = int(d.argmin())
        return k, float(d[k])

    def nearest_edge(self, p):
        if not self.E:
            return None, 1e9, None
        e = np.asarray(sorted(self.E))
        V = np.asarray(self.V)
        a, b = V[e[:, 0]], V[e[:, 1]]
        ab = b - a
        t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(proj - p, axis=1)
        k = int(d.argmin())
        return (int(e[k, 0]), int(e[k, 1])), float(d[k]), proj[k]

    def add_edge(self, a, b):
        self.E.add((a, b))
        self.adj.setdefault(a, set()).add(b)

    def split(self, e, proj):
        a, b = e
        w = self.add_vertex(proj)
        self.E.discard((a, b))
        self.adj[a].discard(b)
        self.add_edge(a, w)
        self.add_edge(w, b)
        return w

    def add_split(self, a, c, snap=1.0, end=1.0):
        """add the edge a -> c, split at every existing vertex that lies on it (within snap, not near the ends); parts that exist already or duplicate a path are skipped"""
        if a == c:
            return
        pa, pc = self.V[a], self.V[c]
        ab = pc - pa
        V = np.asarray(self.V)
        t = ((V - pa) * ab).sum(1) / max(float((ab ** 2).sum()), 1e-12)
        proj = pa + np.clip(t, 0, 1)[:, None] * ab
        d = np.linalg.norm(V - proj, axis=1)
        ok = (t > 0) & (t < 1) & (d < snap) & (np.linalg.norm(V - pa, axis=1) > end) & (np.linalg.norm(V - pc, axis=1) > end)
        ok[[a, c]] = False
        vs = np.nonzero(ok)[0]
        vs = vs[np.argsort(t[vs])]
        chain = [a] + [int(v) for v in vs] + [c]
        for u, w in zip(chain[:-1], chain[1:]):
            if u != w and (u, w) not in self.E and not self.has_path(u, w):
                self.add_edge(u, w)

    def has_path(self, a, b, max_hops=12, slack=3.0):
        """a directed path a -> b of nearly the chord length already exists (the new edge would duplicate it)"""
        chord = np.linalg.norm(self.V[a] - self.V[b])
        frontier, seen = [(a, 0.0)], {a}
        for _ in range(max_hops):
            nxt = []
            for v, L in frontier:
                for w in self.adj.get(v, ()):
                    L2 = L + np.linalg.norm(self.V[w] - self.V[v])
                    if w == b:
                        return L2 <= 1.5 * chord + slack
                    if w not in seen and L2 <= 1.5 * chord + slack:
                        seen.add(w)
                        nxt.append((w, L2))
            frontier = nxt
            if not frontier:
                break
        return False


class FastUnion(Union):
    """Union with the same results and less overhead (loop 16b, speed): the vertices live in a growing array and the edges in a sorted list (the order of the original `sorted(E)` is kept, so ties break the same way);
    no list -> array conversion per call, no np.linalg.norm on 2-vectors."""

    def __init__(self):
        super().__init__()
        self._A = np.zeros((128, 2))
        self._n = 0
        self._es = []

    def add_vertex(self, p):
        p = np.asarray(p, float)
        if self._n == len(self._A):
            self._A = np.concatenate([self._A, np.zeros_like(self._A)])
        self._A[self._n] = p
        self.V.append(p)
        self._n += 1
        return self._n - 1

    def nearest_vertex(self, p):
        if not self._n:
            return -1, 1e9
        d = self._A[: self._n] - p
        d = np.sqrt((d * d).sum(1))
        k = int(d.argmin())
        return k, float(d[k])

    def nearest_edge(self, p):
        if not self._es:
            return None, 1e9, None
        e = np.asarray(self._es)
        V = self._A[: self._n]
        a, b = V[e[:, 0]], V[e[:, 1]]
        ab = b - a
        t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
        proj = a + t[:, None] * ab
        d = proj - p
        d = np.sqrt((d * d).sum(1))
        k = int(d.argmin())
        return (int(e[k, 0]), int(e[k, 1])), float(d[k]), proj[k]

    def add_edge(self, a, b):
        if (a, b) not in self.E:
            bisect.insort(self._es, (a, b))
            self.E.add((a, b))
        self.adj.setdefault(a, set()).add(b)

    def split(self, e, proj):
        a, b = e
        w = self.add_vertex(proj)
        if (a, b) in self.E:
            self.E.discard((a, b))
            del self._es[bisect.bisect_left(self._es, (a, b))]
        self.adj[a].discard(b)
        self.add_edge(a, w)
        self.add_edge(w, b)
        return w

    def add_split(self, a, c, snap=1.0, end=1.0):
        if a == c:
            return
        pa, pc = self.V[a], self.V[c]
        ab = pc - pa
        V = self._A[: self._n]
        t = ((V - pa) * ab).sum(1) / max(float((ab ** 2).sum()), 1e-12)
        proj = pa + np.clip(t, 0, 1)[:, None] * ab
        d = V - proj
        d = np.sqrt((d * d).sum(1))
        da = V - pa
        da = np.sqrt((da * da).sum(1))
        dc = V - pc
        dc = np.sqrt((dc * dc).sum(1))
        ok = (t > 0) & (t < 1) & (d < snap) & (da > end) & (dc > end)
        ok[[a, c]] = False
        vs = np.nonzero(ok)[0]
        vs = vs[np.argsort(t[vs])]
        chain = [a] + [int(v) for v in vs] + [c]
        for u, w in zip(chain[:-1], chain[1:]):
            if u != w and (u, w) not in self.E and not self.has_path(u, w):
                self.add_edge(u, w)

    def has_path(self, a, b, max_hops=12, slack=3.0):
        Va, Vb = self.V[a], self.V[b]
        chord = math.hypot(Va[0] - Vb[0], Va[1] - Vb[1])
        lim = 1.5 * chord + slack
        frontier, seen = [(a, 0.0)], {a}
        for _ in range(max_hops):
            nxt = []
            for v, L in frontier:
                Vv = self.V[v]
                for w in self.adj.get(v, ()):
                    Vw = self.V[w]
                    L2 = L + math.hypot(Vw[0] - Vv[0], Vw[1] - Vv[1])
                    if w == b:
                        return L2 <= lim
                    if w not in seen and L2 <= lim:
                        seen.add(w)
                        nxt.append((w, L2))
            frontier = nxt
            if not frontier:
                break
        return False


def _chain(adj_out, adj_in, s, first, max_hops=14):
    """follow unique successors from s through `first` while the intermediate vertices have in- and out-degree 1; returns the vertex list [s, first, ...] up to the first junction"""
    ch = [s, first]
    while len(ch) < max_hops:
        v = ch[-1]
        if len(adj_out.get(v, ())) != 1 or len(adj_in.get(v, ())) != 1:
            break
        ch.append(next(iter(adj_out[v])))
    return ch


def _pl_dist(P, Q):
    """max distance of the points of path P to the polyline Q"""
    Q = np.asarray(Q)
    out = 0.0
    for p in P:
        a, b = Q[:-1], Q[1:]
        ab = b - a
        t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
        out = max(out, float(np.linalg.norm(p - (a + t[:, None] * ab), axis=1).min()))
    return out


def simplify(V, E, vpi, bubble_r=2.5, spur_len=5.0, rounds=4):
    """graph clean-up of the union: (1) bubbles - two branches that leave a vertex and meet again within a few metres are duplicates, the weaker one (lower mean path validity) is removed;
    (2) spurs - a dead-end branch shorter than spur_len that hangs on a junction is removed.  V [n,2], E set of directed edges, vpi [n]. returns the new edge set."""
    E = set(E)
    for _ in range(rounds):
        out, inn = {}, {}
        for a, b in E:
            out.setdefault(a, set()).add(b)
            inn.setdefault(b, set()).add(a)
        changed = False
        for s in [v for v in out if len(out[v]) >= 2]:
            if s not in out or len(out[s]) < 2:
                continue
            outs = sorted(out[s])
            chains = {b: _chain(out, inn, s, b) for b in outs}
            done = False
            for i in range(len(outs)):
                for j in range(i + 1, len(outs)):
                    c1, c2 = chains[outs[i]], chains[outs[j]]
                    com = [v for v in c1[1:] if v in c2[1:]]
                    if not com:
                        continue
                    m = com[0]
                    p1, p2 = c1[: c1.index(m) + 1], c2[: c2.index(m) + 1]
                    if _pl_dist([V[v] for v in p1[1:-1]], [V[v] for v in p2]) > bubble_r and _pl_dist([V[v] for v in p2[1:-1]], [V[v] for v in p1]) > bubble_r:
                        continue
                    w1 = np.mean([vpi[v] for v in p1]) if len(p1) > 0 else 0
                    w2 = np.mean([vpi[v] for v in p2])
                    drop = p1 if (w1, -len(p1)) < (w2, -len(p2)) else p2
                    for u, w in zip(drop[:-1], drop[1:]):
                        E.discard((u, w))
                    changed = done = True
                    break
                if done:
                    break
        # spurs: leaf chains hanging on a junction
        out, inn = {}, {}
        for a, b in E:
            out.setdefault(a, set()).add(b)
            inn.setdefault(b, set()).add(a)
        for v in list(set(out) | set(inn)):
            if len(out.get(v, ())) == 0 and len(inn.get(v, ())) == 1:  # leaf at the end: walk back
                ch, cur = [v], v
                while len(inn.get(cur, ())) == 1 and len(out.get(cur, ())) <= 1:
                    p = next(iter(inn[cur]))
                    ch.append(p)
                    cur = p
                    if len(out.get(p, ())) >= 2 or len(inn.get(p, ())) != 1:
                        break
                if len(out.get(ch[-1], ())) >= 2 and sum(np.linalg.norm(V[a] - V[b]) for a, b in zip(ch[:-1], ch[1:])) < spur_len:
                    for a, b in zip(ch[1:], ch[:-1]):
                        E.discard((a, b))
                    changed = True
            elif len(inn.get(v, ())) == 0 and len(out.get(v, ())) == 1:  # leaf at the start: walk forward
                ch, cur = [v], v
                while len(out.get(cur, ())) == 1 and len(inn.get(cur, ())) <= 1:
                    n_ = next(iter(out[cur]))
                    ch.append(n_)
                    cur = n_
                    if len(inn.get(n_, ())) >= 2 or len(out.get(n_, ())) != 1:
                        break
                if len(inn.get(ch[-1], ())) >= 2 and sum(np.linalg.norm(V[a] - V[b]) for a, b in zip(ch[:-1], ch[1:])) < spur_len:
                    for a, b in zip(ch[:-1], ch[1:]):
                        E.discard((a, b))
                    changed = True
        if not changed:
            break
    return E


def anchor_junctions(V, E, jmap, thr=0.3, r_edge=3.0, r_free=8.0, ang_hi=100.0):
    """junction anchoring: a peak of the predicted split (merge) heat map above thr is a place where a lane forks (merges). If no vertex with out-degree (in-degree) >= 2 sits within 3 m, the nearest edge
    within r_edge is split at the foot point (the TRUNK) and the nearest free start (free end) within r_free of that point, whose direction is compatible (<= ang_hi against the trunk), is connected to it.
    jmap [2,H,W] probabilities (cells of 0.5 m, +-50 m). returns the new vertex list and edge set."""
    from scipy.ndimage import maximum_filter

    V = [np.asarray(v, float) for v in V]
    E = set(E)
    for kind in (0, 1):
        m = jmap[kind].astype(np.float32)
        pk = (m == maximum_filter(m, size=9)) & (m >= thr)
        ij = np.argwhere(pk)
        if len(ij) == 0:
            continue
        order = np.argsort(-m[pk])
        for i_, j_ in ij[order]:
            p = np.array([(i_ + 0.5) * 0.5 - 50.0, (j_ + 0.5) * 0.5 - 50.0])
            out, inn = {}, {}
            for a, b in E:
                out.setdefault(a, set()).add(b)
                inn.setdefault(b, set()).add(a)
            dist_v = [(float(np.linalg.norm(V[v] - p)), v) for v in set(out) | set(inn)]
            if any(d < 3.0 and ((len(out.get(v, ())) >= 2) if kind == 0 else (len(inn.get(v, ())) >= 2)) for d, v in dist_v):
                continue  # a junction vertex is there already
            best = None
            for (u, w) in E:
                a, b = V[u], V[w]
                ab = b - a
                L2 = float(ab @ ab)
                if L2 < 1e-9:
                    continue
                t = float(np.clip(((p - a) @ ab) / L2, 0.0, 1.0))
                q = a + t * ab
                d = float(np.linalg.norm(p - q))
                if d <= r_edge and (best is None or d < best[0]):
                    best = (d, u, w, q, t)
            if best is None:
                continue
            _, u, w, q, t = best
            e = (V[w] - V[u]) / max(float(np.linalg.norm(V[w] - V[u])), 1e-9)
            own, frontier = {u, w}, [u, w]
            for _ in range(3):
                nxt = []
                for v in frontier:
                    for x in list(out.get(v, ())) + list(inn.get(v, ())):
                        if x not in own:
                            own.add(x)
                            nxt.append(x)
                frontier = nxt
            cand = None
            for s_ in set(out) | set(inn):
                if s_ in own:
                    continue
                if kind == 0 and (len(inn.get(s_, ())) != 0 or len(out.get(s_, ())) != 1):
                    continue
                if kind == 1 and (len(out.get(s_, ())) != 0 or len(inn.get(s_, ())) != 1):
                    continue
                d_ = float(np.linalg.norm(V[s_] - q))
                if d_ > r_free:
                    continue
                nb = next(iter(out[s_])) if kind == 0 else next(iter(inn[s_]))
                dv = (V[nb] - V[s_]) if kind == 0 else (V[s_] - V[nb])
                dv = dv / max(float(np.linalg.norm(dv)), 1e-9)
                if float(np.degrees(np.arccos(np.clip(dv @ e, -1.0, 1.0)))) > ang_hi:
                    continue
                if cand is None or d_ < cand[0]:
                    cand = (d_, s_)
            if cand is None:
                continue
            s_ = cand[1]
            if t < 0.05:
                tgt = u
            elif t > 0.95:
                tgt = w
            else:
                V.append(q)
                tgt = len(V) - 1
                E.discard((u, w))
                E.add((u, tgt))
                E.add((tgt, w))
            E.add((tgt, s_) if kind == 0 else (s_, tgt))
    return np.asarray(V), E


def anchor_junctions_fast(V, E, jmap, thr=0.3, r_edge=3.0, r_free=8.0, ang_hi=100.0, mf=None):
    """anchor_junctions with the same logic and the same tie-breaking, vectorised over the edges and the free ends; the graph structure is built once and again only after a junction was created (loop 16b, speed)"""
    from scipy.ndimage import maximum_filter

    V = [np.asarray(v, float) for v in V]
    E = set(E)

    def build(kind):
        out, inn = {}, {}
        for a, b in E:
            out.setdefault(a, set()).add(b)
            inn.setdefault(b, set()).add(a)
        verts = list(set(out) | set(inn))
        Vv = np.array([V[v] for v in verts]).reshape(-1, 2)
        deg = np.array([len(out.get(v, ())) if kind == 0 else len(inn.get(v, ())) for v in verts])
        El = list(E)
        eu = np.array([u for u, _ in El], int)
        ew = np.array([w for _, w in El], int)
        Va = np.array([V[v] for v in range(len(V))]).reshape(-1, 2)
        EA, EB = Va[eu], Va[ew]
        ab = EB - EA
        L2 = (ab * ab).sum(1)
        fs, fnb, fdir = [], [], []
        for s_ in verts:
            if kind == 0 and (len(inn.get(s_, ())) != 0 or len(out.get(s_, ())) != 1):
                continue
            if kind == 1 and (len(out.get(s_, ())) != 0 or len(inn.get(s_, ())) != 1):
                continue
            nb = next(iter(out[s_])) if kind == 0 else next(iter(inn[s_]))
            dv = (V[nb] - V[s_]) if kind == 0 else (V[s_] - V[nb])
            dv = dv / max(float(np.linalg.norm(dv)), 1e-9)
            fs.append(s_)
            fdir.append(dv)
        return dict(out=out, inn=inn, Vv=Vv, deg=deg, eu=eu, ew=ew, EA=EA, ab=ab, L2=L2, ok=L2 >= 1e-9, fs=np.array(fs, int), fV=np.array([V[s_] for s_ in fs]).reshape(-1, 2), fdir=np.array(fdir).reshape(-1, 2))

    for kind in (0, 1):
        m = jmap[kind].astype(np.float32)
        pk = (m == (mf[kind] if mf is not None else maximum_filter(m, size=9))) & (m >= thr)
        ij = np.argwhere(pk)
        if len(ij) == 0:
            continue
        order = np.argsort(-m[pk])
        st = None
        for i_, j_ in ij[order]:
            p = np.array([(i_ + 0.5) * 0.5 - 50.0, (j_ + 0.5) * 0.5 - 50.0])
            if st is None:
                st = build(kind)
            if len(st["Vv"]):
                dd = st["Vv"] - p
                dd = np.sqrt((dd * dd).sum(1))
                if bool(((dd < 3.0) & (st["deg"] >= 2)).any()):
                    continue  # a junction vertex is there already
            if not len(st["eu"]):
                continue
            t = np.clip(((p - st["EA"]) * st["ab"]).sum(1) / np.where(st["ok"], st["L2"], 1.0), 0.0, 1.0)
            q_all = st["EA"] + t[:, None] * st["ab"]
            dq = p - q_all
            dq = np.sqrt((dq * dq).sum(1))
            valid = st["ok"] & (dq <= r_edge)
            if not valid.any():
                continue
            k = int(np.argmin(np.where(valid, dq, np.inf)))
            u, w, q, tk = int(st["eu"][k]), int(st["ew"][k]), q_all[k], float(t[k])
            out, inn = st["out"], st["inn"]
            e = (V[w] - V[u]) / max(float(np.linalg.norm(V[w] - V[u])), 1e-9)
            own, frontier = {u, w}, [u, w]
            for _ in range(3):
                nxt = []
                for v in frontier:
                    for x in list(out.get(v, ())) + list(inn.get(v, ())):
                        if x not in own:
                            own.add(x)
                            nxt.append(x)
                frontier = nxt
            cand = None
            if len(st["fs"]):
                notown = np.array([int(s_) not in own for s_ in st["fs"]])
                dfree = st["fV"] - q
                dfree = np.sqrt((dfree * dfree).sum(1))
                ang = np.degrees(np.arccos(np.clip(st["fdir"] @ e, -1.0, 1.0)))
                okf = notown & (dfree <= r_free) & (ang <= ang_hi)
                if okf.any():
                    kk = int(np.argmin(np.where(okf, dfree, np.inf)))
                    cand = (float(dfree[kk]), int(st["fs"][kk]))
            if cand is None:
                continue
            s_ = cand[1]
            if tk < 0.05:
                tgt = u
            elif tk > 0.95:
                tgt = w
            else:
                V.append(q)
                tgt = len(V) - 1
                E.discard((u, w))
                E.add((u, tgt))
                E.add((tgt, w))
                st = None
            E.add((tgt, s_) if kind == 0 else (s_, tgt))
            st = None
    return np.asarray(V), E


def _dp_keep(q, eps):
    """indices kept by Douglas-Peucker (iterative) on the polyline q [n,2] with tolerance eps (m); the ends are always kept.
    Same rule as evaluation/sparse_lanes/sparse_metrics._dp_keep (the oracle count vert100_dp); keep the two equal."""
    n = len(q)
    if n <= 2:
        return list(range(n))
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        ab = q[b] - q[a]
        s = np.clip(((q[a + 1:b] - q[a]) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        d = np.linalg.norm(q[a + 1:b] - (q[a] + s[:, None] * ab), axis=1)
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return [int(i) for i in np.nonzero(keep)[0]]


def simplify_dp(g, eps):
    """output stage (E257, decision D20): remove the interior vertices of every polyline that the Douglas-Peucker rule (tolerance eps, m) does not need.
    Kept: both ends of every polyline, every node id that occurs in more than one polyline (junctions, links), and the DP vertices between those anchors, so connectivity,
    the polyline pairs of `edges` and the junction metrics are unchanged and no removed vertex is farther than eps from the new polyline. Returns a new graph dict (pts, edges, nodes)."""
    from collections import Counter

    assert g.get("nodes") is not None and len(g["nodes"]) == len(g["pts"]), "simplify_dp needs a node id list for every polyline"
    cnt = Counter(i for nd in g["nodes"] for i in nd)
    pts, nodes = [], []
    for q, nd in zip(g["pts"], g["nodes"]):
        q = np.asarray(q, np.float64)
        n = len(q)
        assert len(nd) == n, "simplify_dp: node ids and points of a polyline differ in length"
        if n < 3:
            pts.append(q)
            nodes.append(list(nd))
            continue
        anchors = sorted({0, n - 1} | {j for j in range(n) if cnt[nd[j]] > 1})
        keep = set(anchors)
        for a, b in zip(anchors, anchors[1:]):
            keep.update(a + k for k in _dp_keep(q[a:b + 1], eps))
        idx = sorted(keep)
        pts.append(q[idx])
        nodes.append([nd[j] for j in idx])
    return dict(pts=pts, edges=list(g["edges"]), nodes=nodes)


def assemble(r, b, tau, cfg, cov_r=1.0, vtx_r=1.5, pair=None, clean=True, bubble_r=2.5, spur_len=5.0, aux=False, junc_thr=0.0, junc_edge=3.0, junc_free=8.0, junc_ang=100.0, reloc_thr=0.0, reloc_shift=8.0, reloc_tol=0.8, fast=True, tau_low=0.0, fork_prune=0.0, fork_prune_m=-1.0, tau_decay=0.0, dp_eps=0.0):
    """tree -> DAG by path union (nodes in descending path validity): a node within cov_r of an existing edge is snapped onto it (edge split: fork / merge), a node near a vertex reuses it,
    edges that duplicate an existing path are skipped. `pair` = optional learned duplicate suppression (alias of a node to the better node).
    tau_low > 0: hysteresis, a node with validity between tau_low and tau is admitted if its parent is admitted (a lane is extended, no lane is started)."""
    if dp_eps < 0 or (dp_eps > 0 and aux):
        raise ValueError("dp_eps must be >= 0 and cannot be combined with aux=True (the token map of aux refers to the unsimplified vertices)")
    pos = r["pos_rev"] if "pos_rev" in r else r["pos"]
    thr = tau if tau_decay <= 0 else tau * np.exp(-tau_decay * r["level"].astype(np.float64))  # loop 18: the path validity is a cumulative product, so a deeper node is admitted at a lower threshold
    alive = (r["valid"] > 0) & (r["pi"] >= thr)
    if tau_low > 0:
        lowok = (r["valid"] > 0) & (r["pi"] >= tau_low) & ~alive
        if lowok.any():
            alive = alive.copy()
            par_ = r["parent"]
            for i in np.argsort(r["level"], kind="stable"):
                if lowok[i] and par_[i] >= 0 and alive[par_[i]]:
                    alive[i] = True
    idx = np.nonzero(alive)[0]
    empty = dict(pts=[], edges=[], nodes=[])
    if len(idx) == 0:
        return (empty, dict(V=np.zeros((0, 2)), src=[], edges=[])) if aux else empty
    order = idx[np.lexsort((r["level"][idx], -r["pi"][idx]))]
    if fast == 2:  # numba kernels for the nearest queries of the union (loop 18, identical graphs)
        from p18_union_nb import NumbaUnion

        U = NumbaUnion()
    else:
        U = FastUnion() if fast else Union()
    vid = -np.ones(len(pos), int)
    vpi, vsrc = {}, {}
    synth = set()  # vertices created by snapping a node onto an existing edge

    def place(i):
        """vertex of node i: reuse a vertex within vtx_r, else snap onto an existing edge within cov_r (split), else a new vertex. Seeds are placed when a child needs them."""
        if vid[i] < 0 and pair is not None:
            for j, p_ in zip(pair["ki"][i], pair["p"][i]):
                if p_ > 0.5 and alive[j] and vid[j] >= 0 and (r["pi"][j] > r["pi"][i] or (r["pi"][j] == r["pi"][i] and j < i)):
                    vid[i] = vid[j]
                    break
        if vid[i] < 0:
            p = pos[i]
            nv, dv = U.nearest_vertex(p)
            if dv < vtx_r and nv >= 0:
                vid[i] = nv
            else:
                e, de, proj = U.nearest_edge(p)
                if e is not None and de < cov_r:
                    vid[i] = U.split(e, proj)
                    synth.add(vid[i])
                else:
                    vid[i] = U.add_vertex(p)
        if float(r["pi"][i]) >= vpi.get(vid[i], 0.0):
            vsrc[vid[i]] = int(i)
        vpi[vid[i]] = max(vpi.get(vid[i], 0.0), float(r["pi"][i]))
        return vid[i]

    for i in order:
        par = int(r["parent"][i])
        if par < 0:
            continue
        pa = place(par)
        ci = place(i)
        a, c = (pa, ci) if r["flag"][i] > 0 else (ci, pa)
        if a != c and (a, c) not in U.E and not U.has_path(a, c):
            U.add_split(a, c, cov_r)
    for (i, j, pl) in r.get("link_p", []):
        if pl >= 0.5 and alive[i] and alive[j]:
            a, c = place(i), place(j)
            if a != c and (a, c) not in U.E and not U.has_path(a, c):
                U.add_split(a, c, cov_r)
    E = U.E
    if clean and E:
        vp = np.array([vpi.get(v, 0.5) for v in range(len(U.V))])
        E = simplify(np.asarray(U.V), E, vp, bubble_r, spur_len)
    if E and synth:  # snap vertices that only pass a straight chord through (in = out = 1) are not needed
        out, inn = {}, {}
        for a, c in E:
            out.setdefault(a, set()).add(c)
            inn.setdefault(c, set()).add(a)
        for v in sorted(synth):
            if len(inn.get(v, ())) == 1 and len(out.get(v, ())) == 1:
                a, c = next(iter(inn[v])), next(iter(out[v]))
                E.discard((a, v)), E.discard((v, c))
                out[a].discard(v), inn[c].discard(v)
                del inn[v], out[v]
                if a != c and (a, c) not in E:
                    E.add((a, c))
                    out[a].add(c)
                    inn.setdefault(c, set()).add(a)
    Vall = np.asarray(U.V)
    jm, mf = r.get("junc"), None
    if fast and jm is not None and E and (junc_thr > 0 or reloc_thr > 0):
        from scipy.ndimage import maximum_filter

        jm = [jm[k].astype(np.float32) for k in (0, 1)]  # one conversion and one maximum filter per heat map for both steps
        mf = [maximum_filter(m_, size=9) for m_ in jm]
    if junc_thr > 0 and E and "junc" in r:
        if fast:
            Vall, E = anchor_junctions_fast(Vall, E, jm, junc_thr, junc_edge, junc_free, junc_ang, mf=mf)
        else:
            Vall, E = anchor_junctions(Vall, E, r["junc"], junc_thr, junc_edge, junc_free, junc_ang)
    if reloc_thr > 0 and E and "junc" in r:
        Vall, E = relocate_forks(Vall, E, jm if fast else r["junc"], reloc_thr, reloc_shift, reloc_tol, mf=mf)
    if (fork_prune > 0 or fork_prune_m > 0) and E and "junc" in r:  # loop 18: un-link forks without support of the junction head (geometry unchanged)
        jj = jm if (fast and jm is not None) else [np.asarray(r["junc"][k], np.float32) for k in (0, 1)]
        Vall, E = prune_forks(Vall, E, jj, fork_prune, fork_prune if fork_prune_m < 0 else fork_prune_m)
    if not E:
        return (empty, dict(V=np.zeros((0, 2)), src=[], edges=[])) if aux else empty
    used = sorted({v for e in E for v in e})
    remap = {v: k for k, v in enumerate(used)}
    V = Vall[used]
    ve = [(remap[a], remap[c]) for a, c in sorted(E)]
    g = C.graph_from_vertices(V, ve, legacy_cut=True)  # D24 open item: the consumer / route heads were trained on this piece cut; the
    # fixed cut (legacy_cut=False) moves F1o2o +0.001, dup_frac up to +0.005, vert100 of the *_dp names -0.16 and was not yet measured on route success
    if dp_eps > 0:  # output stage (E257 / D20): drop interior vertices of straight runs, geometry kept within dp_eps; off by default (guard at the top: aux and dp_eps exclude each other)
        g = simplify_dp(g, dp_eps)
    if aux:  # vertex positions, the node whose token a vertex inherits (the highest path validity among the nodes snapped onto it) and the vertex-level edges
        return g, dict(V=V, src=[vsrc.get(v, -1) for v in used], edges=ve)
    return g


def to_numpy(res, i, keys=("pos", "th_travel", "flag", "pi", "nu", "valid", "parent", "level", "pos_rev")):
    out = {}
    for k in keys:
        if k in res:
            out[k] = res[k][i].detach().float().cpu().numpy()
    out["parent"] = out["parent"].astype(np.int64)
    if "junc_logit" in res:
        out["junc"] = torch.sigmoid(res["junc_logit"][i].float()).detach().cpu().numpy().astype(np.float16)
    if "link_ki" in res:  # L: links between forward and backward ends of different pieces (global node indices)
        off = res["link_off"]
        p = torch.sigmoid(res["link_logit"][i]).detach().cpu().numpy() * res["link_mask"][i].cpu().numpy()
        ki = res["link_ki"][i].cpu().numpy()
        out["link_p"] = [(off + a, off + int(ki[a, c]), float(p[a, c])) for a in range(ki.shape[0]) for c in range(ki.shape[1]) if p[a, c] > 0.05]
    return out


@torch.no_grad()
def predict(builder, ds, ks, taus=TAUS, bs=8, workers=4, dev="cuda", pair_p=True, raw=None):
    """runs the builder on the frames ks of dataset ds; returns {tau: [graphs]} and the ground truths (raw: optional list that receives the per-frame node arrays)"""
    builder.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, collate_fn=PD.collate)
    graphs = {t: [] for t in taus}
    gts = []
    t0 = time.time()
    for b in dl:
        bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        prior = None
        if builder.cfg.temporal:
            import p16_time as PT_

            prior = PT_.prior_from_prev(builder, bb, train=False)
        with torch.autocast("cuda", dtype=torch.float16):
            res = builder(bb["x"].float(), gt=None, share=0.0, prior=prior)
        pair = None
        if builder.cfg.dedup == "learned" and pair_p:
            _, pr = PL.dup_learned(builder, res, bb)
            pair = dict(ki=pr["ki"].cpu().numpy(), p=torch.sigmoid(pr["logit"]).float().cpu().numpy())
        for i in range(bb["x"].shape[0]):
            r = to_numpy(res, i)
            if raw is not None:
                raw.append(r)
            for t in taus:
                graphs[t].append(assemble(r, None, t, builder.cfg, pair=None if pair is None else dict(ki=pair["ki"][i], p=pair["p"][i])))
        gts += [ds.gt[ds.rows[int(k)]] for k in b["k"]] if hasattr(ds, "gt") else []
    return graphs, gts, time.time() - t0


def evaluate_sweep(graphs, gts):
    """metric card per threshold + frontier scalars; headline = the threshold with the best composite S"""
    cards = {}
    for t, g in graphs.items():
        m = C.card(g, gts)
        m["tau"] = t
        cards[t] = m
    curve = [dict(thr=t, vert100=m["vert100"], **{k: m[k] for k in ("F1@1.0", "R@1.0", "false_len_frac")}) for t, m in cards.items()]
    fr = C.frontier_scalars(curve)
    sane = [m for m in cards.values() if m["R@1.0"] >= 0.3] or list(cards.values())  # a near-empty graph must not win by its sparsity
    best = max(sane, key=lambda m: m["S"])
    return dict(best=best, at05=cards.get(0.5), sweep={str(t): {k: m[k] for k in ("S", "F1@1.0", "F1@0.5", "J", "Sp", "false_len_frac", "vert100", "R@1.0", "P@1.0", "lanes_found", "split6_F1", "merge6_F1")} for t, m in cards.items()}, **fr)


def _peaks(m, thr, size=9, mf=None):
    """local maxima >= thr of a junction heat map [H, W] (cells of 0.5 m, +-50 m) as ego-frame points; mf = its maximum filter if it was computed already"""
    from scipy.ndimage import maximum_filter

    pk = (m == (mf if mf is not None else maximum_filter(m, size=size))) & (m >= thr)
    ij = np.argwhere(pk)
    if not len(ij):
        return np.zeros((0, 2))
    return np.stack([(ij[:, 0] + 0.5) * 0.5 - 50.0, (ij[:, 1] + 0.5) * 0.5 - 50.0], 1).astype(np.float64)


def prune_forks(V, E, jmap, thr_split, thr_merge):
    """loop 18 (K190): un-link the forks the junction head does not support. A split (merge) vertex whose 3 x 3 junction heat is below the threshold keeps ONE successor (predecessor) - the one that continues
    the incoming (outgoing) direction best - and every other branch starts (ends) at a copy of the vertex: the geometry is unchanged, only the topology (the false fork) disappears.
    Validation of the idea: p18_fork_features.py (heat separates true from false forks, AUROC 0.85 / 0.89)."""
    V = [np.asarray(v, float) for v in V]
    E = set(E)

    def heat(kind, p):
        m = jmap[kind]
        i = int(np.clip(round((p[0] + 50.0) / 0.5 - 0.5), 1, m.shape[0] - 2))
        j = int(np.clip(round((p[1] + 50.0) / 0.5 - 0.5), 1, m.shape[1] - 2))
        return float(m[i - 1 : i + 2, j - 1 : j + 2].max())

    for kind, thr in ((0, thr_split), (1, thr_merge)):
        if thr <= 0:
            continue
        out, inn = {}, {}
        for a, b in E:
            out.setdefault(a, set()).add(b)
            inn.setdefault(b, set()).add(a)
        br, tr = (out, inn) if kind == 0 else (inn, out)  # branch side / trunk side of the fork vertex
        for v in sorted(x for x in br if len(br[x]) >= 2):
            if heat(kind, V[v]) >= thr:
                continue
            nb = sorted(br[v])
            ref = None
            for t in sorted(tr.get(v, ())):  # direction of the trunk through the vertex
                d = (V[v] - V[t]) if kind == 0 else (V[t] - V[v])
                ref = d / max(float(np.linalg.norm(d)), 1e-9)
                break
            if ref is None:
                ref = sum((((V[w] - V[v]) if kind == 0 else (V[v] - V[w])) / max(float(np.linalg.norm(V[w] - V[v])), 1e-9)) for w in nb)
                ref = ref / max(float(np.linalg.norm(ref)), 1e-9)
            cs = [float((((V[w] - V[v]) if kind == 0 else (V[v] - V[w])) / max(float(np.linalg.norm(V[w] - V[v])), 1e-9)) @ ref) for w in nb]
            main = nb[int(np.argmax(cs))]
            for w in nb:
                if w == main:
                    continue
                V.append(V[v].copy())
                v2 = len(V) - 1
                if kind == 0:
                    E.discard((v, w))
                    E.add((v2, w))
                else:
                    E.discard((w, v))
                    E.add((w, v2))
    return np.asarray(V), E


def relocate_forks(V, E, jmap, thr=0.5, max_shift=8.0, tol=0.8, near=2.0, mf=None):
    """fork relocation (export step): a fork vertex (out-degree >= 2; for the merge map: a merge vertex, in-degree >= 2) of a sparse graph sits where the branches have separated, usually one step after the annotated fork.
    If a strong peak of the split (merge) heat map lies within `near` of the stem (the trunk) at most `max_shift` before the vertex, the vertex moves to the foot point of the peak: the stem ends there and the branch edges start there.
    Done only if the straight edges from the new vertex stay within `tol` of the replaced path (shallow forks)."""
    V = [np.asarray(v, float) for v in V]
    E = set(E)
    for kind in (0, 1):
        P = _peaks(jmap[kind].astype(np.float32), thr, mf=None if mf is None else mf[kind])
        if len(P) == 0:
            continue
        out, inn = {}, {}
        for a, b in E:
            out.setdefault(a, set()).add(b)
            inn.setdefault(b, set()).add(a)
        br, tr = (out, inn) if kind == 0 else (inn, out)  # branches / trunk side of the fork vertex
        for v in [x for x in set(out) | set(inn) if len(br.get(x, ())) >= 2 and len(tr.get(x, ())) == 1]:
            if len(br.get(v, ())) < 2 or len(tr.get(v, ())) != 1:
                continue  # changed by an earlier relocation
            chain, cur, ln = [v], v, 0.0
            while ln < max_shift + 4.0:
                nx_ = tr.get(cur, ())
                if len(nx_) != 1:
                    break
                nx = next(iter(nx_))
                if nx in chain or len(chain) > 40:  # a cycle or a zero-length chain of coincident vertices
                    break
                if cur != v and len(br.get(cur, ())) != 1:
                    break
                ln += float(np.linalg.norm(V[cur] - V[nx]))
                chain.append(nx)
                cur = nx
            if len(chain) < 2:
                continue
            poly = np.array([V[c] for c in chain])
            seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
            cs = np.concatenate([[0.0], np.cumsum(seg)])
            best = None
            cand_pt, cand_s, cand_k, cand_f = [], [], [], []
            for k in range(len(chain) - 1):
                n_ = max(int(seg[k] / 0.5), 1)
                fr = np.linspace(0.0, 1.0, n_ + 1)
                cand_pt.append(poly[k] + fr[:, None] * (poly[k + 1] - poly[k]))
                cand_s.append(cs[k] + fr * seg[k])
                cand_k.append(np.full(n_ + 1, k))
                cand_f.append(fr)
            cand_pt, cand_s, cand_k, cand_f = np.concatenate(cand_pt), np.concatenate(cand_s), np.concatenate(cand_k), np.concatenate(cand_f)
            sel = np.nonzero((cand_s >= 1.0) & (cand_s <= max_shift))[0]
            if len(sel):
                dm = cand_pt[sel][:, None, :] - P[None]
                dd = np.sqrt((dm * dm).sum(2)).min(1)
                okd = dd <= near
                if okd.any():
                    j_ = int(np.argmin(np.where(okd, dd, np.inf)))
                    i_ = sel[j_]
                    best = (float(dd[j_]), int(cand_k[i_]), float(cand_f[i_]), cand_pt[i_])
            if best is None:
                continue
            _, k, f, t = best
            bnb = sorted(br[v])
            ok = True
            for b_ in bnb:  # the replaced path t -> chain[k] -> ... -> v -> b_ against the straight edge t - b_
                path = [t] + [V[c] for c in chain[k::-1]] + [V[b_]]
                seg_ = V[b_] - t
                L2 = float(seg_ @ seg_)
                for pnt in path:
                    u_ = float(np.clip(((pnt - t) @ seg_) / max(L2, 1e-9), 0.0, 1.0))
                    if float(np.linalg.norm(pnt - (t + u_ * seg_))) > tol:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            anchor = chain[k + 1]
            if np.linalg.norm(V[anchor] - t) < 0.5:
                tv = anchor
            else:
                V.append(t)
                tv = len(V) - 1
            for j in range(k + 1):  # remove the trunk part between the old vertex and t, and the old branch edges
                a, b = (chain[j + 1], chain[j]) if kind == 0 else (chain[j], chain[j + 1])
                E.discard((a, b))
            for b_ in bnb:
                E.discard((v, b_) if kind == 0 else (b_, v))
            if tv != anchor:
                E.add((anchor, tv) if kind == 0 else (tv, anchor))
            for b_ in bnb:
                E.add((tv, b_) if kind == 0 else (b_, tv))
            out, inn = {}, {}
            for a, b in E:
                out.setdefault(a, set()).add(b)
                inn.setdefault(b, set()).add(a)
            br, tr = (out, inn) if kind == 0 else (inn, out)
    return np.asarray(V), E
