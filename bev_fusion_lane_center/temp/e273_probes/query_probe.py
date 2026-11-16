"""one-off (scratch): representation oracle of the grid-query design on the VAL lane GT (graph_gt_val.pkl = make_graph_gt.py; the lane GT chat's cross-check: within 0.3 m).

Design checked:
  nodes  = per 2 m cell (half-open) and per piece the GT point nearest the cell centre; the piece ends (start / end) win their cell ("events")
           variant "all": every piece; variant "lanes": connector pieces are suppressed (inside intersections there are no nodes)
  links  = consecutive nodes along a piece, and across piece ends (through suppressed connectors); distance <= 8 m "short", else "long"
  decode = cubic Hermite segment per link (handles chord / 3; long links with two fitted handle scales) -> compared with the GT points
  sections (ids by the id rule: pieces joined while 1 in / 1 out) = absorbing walk backwards on the link matrix (dense solve), with noisy probabilities
VAL scenes only, 1 frame in 4.
"""
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

H = os.path.expanduser("~")
VAL = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42}
G = pickle.load(open(f"{H}/exp_gcar_b123/graph_gt_val.pkl", "rb"))
sc = np.load(f"{H}/exp_gcar_b123/poses_val.npz")["scene"]
CELL, HALF, SHORT = 2.0, 50.0, 8.0
rng = np.random.RandomState(0)


def pieces_of(g):
    out = []
    for pi, (p, kd) in enumerate(zip(g["pts"], g["kind"])):
        p = np.asarray(p, float)
        if len(p) < 2:
            out.append(None)
            continue
        d = np.gradient(p, axis=0)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        out.append((p, d, kd))
    return out


def nodes_of(P, suppress, g=None, keep_open=False):
    """-> list of dict(piece, k, pos, hdg, ev) sorted by (piece, k)"""
    nodes = []
    for pi, pc in enumerate(P):
        if pc is None:
            continue
        p, d, kd = pc
        if suppress and kd == "connector":
            if not (keep_open and g is not None):
                continue
        cell = np.clip(np.floor((p + HALF) / CELL).astype(int), 0, int(2 * HALF / CELL) - 1)
        key = cell[:, 0] * 1000 + cell[:, 1]
        best = {}
        for k in range(len(p)):
            ctr = (cell[k] + 0.5) * CELL - HALF
            sc_ = -1.0 if k in (0, len(p) - 1) else float(np.linalg.norm(p[k] - ctr))
            if key[k] not in best or sc_ < best[key[k]][0]:
                best[key[k]] = (sc_, k)
        for _, k in sorted(best.values(), key=lambda e: e[1]):
            if suppress and kd == "connector":
                has_pred = any(j == pi for _, j in g["edges"])
                has_succ = any(i == pi for i, _ in g["edges"])
                is_ev = (k == 0 and (g["start_open"][pi] or not has_pred)) or (k == len(p) - 1 and (g["end_open"][pi] or not has_succ))
                if not is_ev:
                    continue
            nodes.append(dict(piece=pi, k=k, pos=p[k], hdg=d[k], ev=k in (0, len(p) - 1), cell=key[0] if False else 0))
        for nd in nodes:
            if nd["piece"] == pi:
                nd["cell"] = int(key[nd["k"]])
    return nodes


def sections_of(n_pieces, edges):
    outd, ind = np.zeros(n_pieces, int), np.zeros(n_pieces, int)
    for i, j in edges:
        outd[i] += 1
        ind[j] += 1
    par = list(range(n_pieces))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for i, j in edges:
        if outd[i] == 1 and ind[j] == 1:
            par[f(i)] = f(j)
    return [f(i) for i in range(n_pieces)], outd, ind


def links_of(nodes, P, edges):
    by_piece = {}
    for a, nd in enumerate(nodes):
        by_piece.setdefault(nd["piece"], []).append(a)
    succ = {}
    for i, j in edges:
        succ.setdefault(i, []).append(j)
    L = {}
    for pi, lst in by_piece.items():
        for a, b in zip(lst[:-1], lst[1:]):
            L[(a, b)] = ()
        # across the piece end: through pieces without nodes (their points are what a long link has to reproduce)
        stack, seen = [(s, 0, ()) for s in succ.get(pi, [])], set()
        while stack:
            q, dep, path = stack.pop()
            if (q, dep) in seen or dep > 8:
                continue
            seen.add((q, dep))
            if q in by_piece:
                key = (lst[-1], by_piece[q][0])
                if key not in L or len(path) < len(L[key]):
                    L[key] = path
            else:
                stack += [(s, dep + 1, path + (q,)) for s in succ.get(q, [])]
    return L


def hermite(p0, h0, p1, h1, a=1 / 3, b=1 / 3, n=None):
    c = np.linalg.norm(p1 - p0)
    n = n or max(int(c / 0.5) + 2, 3)
    s = np.linspace(0, 1, n)[:, None]
    P1, P2 = p0 + a * c * h0, p1 - b * c * h1
    return (1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * P1 + 3 * (1 - s) * s ** 2 * P2 + s ** 3 * p1


def run(variant):
    suppress = variant in ("lanes", "lanes_win")
    st = dict(frames=0, nodes=0, events=0, cells2=0, cells3=0, cells=0, links=0, short=0, long=0, cand_short=0, miss_short=0, cand_long=0, hit_long=0,
              cone_long=0, nlong_entries=0, ent_cand=0, deg_out=np.zeros(6, int), deg_in=np.zeros(6, int))
    rec, prec_, fit_a, kind_rec = [], [], [], {}
    mem_acc = {}
    nf = 0
    for k, g in enumerate(G):
        if int(sc[k]) not in VAL:
            continue
        nf += 1
        if nf % 4:
            continue
        P = pieces_of(g)
        nodes = nodes_of(P, suppress, g, keep_open=variant == "lanes_win")
        if len(nodes) < 3:
            continue
        N = len(nodes)
        st["frames"] += 1
        st["nodes"] += N
        st["events"] += sum(n_["ev"] for n_ in nodes)
        cells = {}
        for n_ in nodes:
            cells[n_["cell"]] = cells.get(n_["cell"], 0) + 1
        st["cells"] += len(cells)
        st["cells2"] += sum(v >= 2 for v in cells.values())
        st["cells3"] += sum(v >= 3 for v in cells.values())
        LD = links_of(nodes, P, g["edges"])
        L = sorted(LD)
        pos = np.array([n_["pos"] for n_ in nodes])
        hd = np.array([n_["hdg"] for n_ in nodes])
        dist = np.linalg.norm(pos[:, None] - pos[None], axis=2)
        st["links"] += len(L)
        outn, inn = np.zeros(N, int), np.zeros(N, int)
        for a, b in L:
            outn[a] += 1
            inn[b] += 1
        for x in outn:
            st["deg_out"][min(x, 5)] += 1
        for x in inn:
            st["deg_in"][min(x, 5)] += 1
        # candidates
        fwd = ((pos[None] - pos[:, None]) * hd[:, None]).sum(2) > 0
        cs = (dist <= SHORT) & fwd & ~np.eye(N, dtype=bool)
        st["cand_short"] += int(cs.sum())
        short = [(a, b) for a, b in L if dist[a, b] <= SHORT]
        long_ = [(a, b) for a, b in L if dist[a, b] > SHORT]
        st["short"] += len(short)
        st["long"] += len(long_)
        st["miss_short"] += sum(not cs[a, b] for a, b in short)
        ents = sorted({a for a, _ in long_})
        st["nlong_entries"] += len(ents)
        exits = sorted({b for _, b in long_})
        cone = (((pos[None] - pos[:, None]) / np.maximum(dist, 1e-9)[..., None] * hd[:, None]).sum(2) > np.cos(np.deg2rad(75))) & (dist <= 100) & (dist > SHORT)
        for a in ents:
            st["ent_cand"] += int(cone[a, exits].sum())          # candidates if a perfect exit classifier preselects the exit nodes
            st["cand_long"] += int(cone[a].sum())                 # candidates if every node is a candidate
        st["hit_long"] += sum(cone[a, b] for a, b in long_)
        # Hermite decode
        gt_all = np.concatenate([pc[0] for pc in P if pc is not None])
        tree_gt = cKDTree(gt_all)
        dec = []
        kind_pts = {kd_: np.concatenate([pc[0] for pc in P if pc is not None and pc[2] == kd_] or [np.zeros((0, 2))]) for kd_ in ("lane", "connector")}
        for a, b in L:
            if dist[a, b] <= SHORT:
                dec.append(hermite(pos[a], hd[a], pos[b], hd[b]))
            else:
                tgt = np.concatenate([P[q][0] for q in LD[(a, b)] if P[q] is not None] or [gt_all])
                tt = cKDTree(tgt)
                best, bc = 1e9, None
                for sa in (0.1, 0.25, 0.4, 0.55, 0.7):
                    for sb in (0.1, 0.25, 0.4, 0.55, 0.7):
                        cur = hermite(pos[a], hd[a], pos[b], hd[b], sa, sb)
                        m = tt.query(cur)[0].mean() + 0.2 * cKDTree(cur).query(tgt)[0].mean()
                        if m < best:
                            best, bc = m, cur
                dec.append(bc)
        dec = np.concatenate(dec) if dec else np.zeros((0, 2))
        if len(dec):
            td = cKDTree(dec)
            rec.append(float((td.query(gt_all)[0] < 0.3).mean()))
            for kd_ in ("lane", "connector"):
                if len(kind_pts[kd_]):
                    kind_rec.setdefault(kd_, []).append(float((td.query(kind_pts[kd_])[0] < 0.3).mean()))
            prec_.append(float((tree_gt.query(dec)[0] < 0.3).mean()))
        # sections: absorbing walk backwards, noisy probabilities
        sec_p, _, _ = sections_of(len(P), g["edges"])
        sec = np.array([sec_p[n_["piece"]] for n_ in nodes])
        prev = -np.ones(N, int)
        for a, b in L:
            if sec[a] == sec[b]:
                prev[b] = a
        # the first node of each section = no in-section predecessor
        first = prev < 0
        start_of = np.arange(N)
        for n_ in range(N):
            m = n_
            while prev[m] >= 0:
                m = prev[m]
            start_of[n_] = m
        depth = np.zeros(N, int)
        for n_ in range(N):
            m, dd = n_, 0
            while prev[m] >= 0:
                m = prev[m]
                dd += 1
            depth[n_] = dd
        for noise in ("clean", "noisy"):
            p_true, s_true, s_false = (1.0, 1.0, 0.0) if noise == "clean" else (0.97, 0.95, 0.01)
            Pb = np.zeros((N, N))
            for n_ in range(N):
                cand = np.where((dist[n_] <= SHORT) & ~fwd[n_] & (np.arange(N) != n_))[0]       # behind n_ within 8 m
                if prev[n_] >= 0:
                    Pb[n_, prev[n_]] = p_true
                    rest = [c for c in cand if c != prev[n_]]
                    if rest:
                        Pb[n_, rest] = (1 - p_true) / len(rest)
                    else:
                        Pb[n_, prev[n_]] = 1.0
                elif len(cand):
                    Pb[n_, cand] = 1.0 / len(cand)
            s = np.where(first, s_true, s_false)
            Q = (1 - s)[:, None] * Pb
            M = np.linalg.solve(np.eye(N) - Q, np.diag(s))
            M /= np.maximum(M.sum(1, keepdims=True), 1e-12)
            ok = M.argmax(1) == start_of
            for d_ in (0, 1, 3, 10, 30):
                msk = depth >= d_
                if noise == "noisy" and msk.any():
                    mem_acc.setdefault(d_, []).append((ok[msk].sum(), msk.sum()))
    nfr = st["frames"]
    print(f"--- variant {variant}: {nfr} frames")
    print(f"nodes per frame {st['nodes'] / nfr:.0f} (events {st['events'] / nfr:.0f}); occupied cells {st['cells'] / nfr:.0f}, cells with >= 2 nodes {st['cells2'] / st['cells']:.3f}, >= 3 {st['cells3'] / st['cells']:.4f}")
    print(f"links per frame {st['links'] / nfr:.0f}: short (<= {SHORT:.0f} m) {st['short'] / nfr:.0f}, long {st['long'] / nfr:.1f}")
    print(f"out-degree of nodes: " + ", ".join(f"{i}{'+' if i == 5 else ''}: {v / st['nodes']:.3f}" for i, v in enumerate(st['deg_out'])))
    print(f"in-degree of nodes:  " + ", ".join(f"{i}{'+' if i == 5 else ''}: {v / st['nodes']:.3f}" for i, v in enumerate(st['deg_in'])))
    print(f"short candidates per node (forward, <= {SHORT:.0f} m) {st['cand_short'] / st['nodes']:.1f}; true short link missed by the candidates {st['miss_short'] / max(st['short'], 1):.4f}")
    if st["nlong_entries"]:
        print(f"long links: entries per frame {st['nlong_entries'] / nfr:.1f}; true partner in the 100 m / 75 deg cone {st['hit_long'] / max(st['long'], 1):.3f}; candidates per entry: all nodes {st['cand_long'] / st['nlong_entries']:.0f}, exit nodes only {st['ent_cand'] / st['nlong_entries']:.1f}")
    print(f"Hermite decode of the GT links (long links: handles fitted from 5 x 5): GT points within 0.3 m of the decoded curves {np.mean(rec):.4f}, decoded points within 0.3 m of the GT {np.mean(prec_):.4f}")
    print("   recall by GT kind: " + ", ".join(f"{k_}: {np.mean(v):.4f}" for k_, v in kind_rec.items()))
    print("section membership by the absorbing walk (noisy: true link 0.97, start 0.95, false start 0.01), argmax correct, nodes at least d nodes after the section start:")
    print("   " + ", ".join(f"d>={d_}: {sum(a for a, _ in v) / sum(b for _, b in v):.3f} (n {sum(b for _, b in v)})" for d_, v in sorted(mem_acc.items())))


if __name__ == "__main__":
    for v in sys.argv[1:] or ("all", "lanes"):
        run(v)
