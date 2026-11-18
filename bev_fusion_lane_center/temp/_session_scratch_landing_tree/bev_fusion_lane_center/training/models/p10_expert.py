"""Loop 10: corrective expert for the learned trace network (arm A of the sparse-graph proposal). Ground truth only, no network.
A head moves along a lane graph; the expert tells, for any state (position p, heading theta, direction flag) and any set of 'already emitted' ground-truth samples,
what the next step should be: heading bin + residual, step class {2, 4, 6 m} + residual, event (continue / end / fork / merge / border), child angle, merge target.
The graph is the ground-truth piece graph (pieces in travel direction, edges end(i) = start(j)); direction flag +1 moves with the travel direction, -1 against it (reversed graph).
Usage: imported by p10_wave.py / p10_data.py."""
import math

import numpy as np
from scipy.spatial import cKDTree

import graph_eval as GE

STEPS = (2.0, 4.0, 6.0)
BIN = math.radians(11.25)
N_BIN = 9
CONT, END, FORK, MERGE, BORDER = 0, 1, 2, 3, 4
DS = 0.5
CLIP = 48.0
KMAX = 12  # samples (6 m) a step may span


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def ang(v):
    return math.atan2(v[1], v[0])


def unit(th):
    return np.array([math.cos(th), math.sin(th)])


class ExpertGraph:
    def __init__(self, gt, rev=False):
        n = len(gt["pts"])
        self.n, self.rev = n, rev
        self.pts, self.arc = [], []
        for p in gt["pts"]:
            q = GE.resample(np.asarray(p, np.float64), DS) if len(p) >= 2 else np.zeros((0, 2))
            if len(q) < 2:
                q = np.zeros((0, 2))
            if rev:
                q = q[::-1].copy()
            self.pts.append(q)
            self.arc.append(np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))] if len(q) else np.zeros(0))
        self.succ = [[] for _ in range(n)]
        self.pred = [[] for _ in range(n)]
        for i, j in gt["edges"]:
            if rev:
                i, j = j, i
            if len(self.pts[i]) and len(self.pts[j]):
                self.succ[i].append(j)
                self.pred[j].append(i)
        oe = gt.get("start_open" if rev else "end_open", [False] * n)
        self.open_end = [bool(x) for x in oe]
        self.owner = [min(self.pred[j]) if self.pred[j] else -1 for j in range(n)]
        allp = [p for p in self.pts if len(p)]
        self.has = len(allp) > 0
        if self.has:
            self.allp = np.concatenate(allp)
            self.own = np.concatenate([np.full(len(p), i) for i, p in enumerate(self.pts) if len(p)])
            self.idx = np.concatenate([np.arange(len(p)) for p in self.pts if len(p)])
            self.tree = cKDTree(self.allp)
        self.tan = []
        for p in self.pts:
            if len(p) >= 2:
                t = np.gradient(p, axis=0)
                t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
            else:
                t = np.zeros((0, 2))
            self.tan.append(t)

    # ---------------------------------------------------------------- geometry
    def remaining(self, i, ix):
        return (len(self.pts[i]) - 1 - ix) * DS

    def project(self, p, theta=None, rad=3.0):
        """nearest graph sample (piece, index, distance), preferring heading agreement and, at ties, the piece with more length ahead."""
        if not self.has:
            return None
        ds, js = self.tree.query(p, k=min(10, len(self.allp)), distance_upper_bound=rad)
        best, bs = None, 1e9
        for d, j in zip(np.atleast_1d(ds), np.atleast_1d(js)):
            if not np.isfinite(d) or j >= len(self.allp):
                continue
            i, ix = int(self.own[j]), int(self.idx[j])
            sc = d - 0.004 * min(self.remaining(i, ix), 6.0)
            if theta is not None:
                sc += 1.2 * (1.0 - float(np.dot(self.tan[i][ix], unit(theta))))
            if sc < bs:
                best, bs = (i, ix, float(d)), sc
        return best

    def straightest(self, i, cands):
        """successor with the smallest turn relative to the end tangent of piece i"""
        t0 = self.tan[i][-1]
        return max(cands, key=lambda j: float(np.dot(t0, self.tan[j][0])))

    def ahead(self, i, ix, dist, em=None):
        """position at path distance dist from sample (i, ix) along the straightest (untraced) successors; (point, piece, index) or None if the path ends"""
        n = int(round(dist / DS))
        while True:
            if ix + n < len(self.pts[i]):
                return self.pts[i][ix + n], i, ix + n
            n -= len(self.pts[i]) - 1 - ix
            c = self.succ[i]
            if em is not None:
                u = [j for j in c if not em[j][:3].any()]
                c = u if u else c
            if not c:
                return None
            i, ix = self.straightest(i, c), 0

    def own_future(self, i, ix, length, dmin=0.0):
        """samples (piece, from, to) ahead of (i, ix) over `length` m through owned successors (the head traces them itself), skipping the first dmin m"""
        out = []
        stack = [(i, ix + 1 + int(round(dmin / DS)), length - dmin)]
        seen = set()
        while stack:
            pc, a, rem = stack.pop()
            if rem <= 0 or pc in seen:
                continue
            seen.add(pc)
            b = min(len(self.pts[pc]), a + int(round(rem / DS)) + 1)
            if b > a:
                out.append((pc, a, b))
            used = (b - a) * DS
            if b >= len(self.pts[pc]):
                for j in self.succ[pc]:
                    if self.owner[j] == pc:
                        stack.append((j, 0, rem - used))
        return out

    def back_path(self, i, ix, dist):
        """positions behind (i, ix) at path distance dist (through the straightest predecessor); None when the path starts"""
        n = int(round(dist / DS))
        while True:
            if ix - n >= 0:
                return self.pts[i][ix - n]
            n -= ix
            if not self.pred[i]:
                return None
            j = max(self.pred[i], key=lambda x: float(np.dot(self.tan[i][0], self.tan[x][-1])))
            i, ix = j, len(self.pts[j]) - 1

    # ---------------------------------------------------------------- expert
    def targets(self, p, theta, em, noise_lat=0.0, noise_lon=0.0):
        """Expert action for the state; em[piece] = bool array over the samples of that piece (already emitted by someone else).
        Returns None when p is not on the graph. noise_*: the block-2 candidate is displaced by these amounts (offset target = -noise_lat)."""
        pr = self.project(p, theta)
        if pr is None:
            return None
        i, ix, dist = pr
        n_i = len(self.pts[i])
        k_end = n_i - 1 - ix
        if k_end == 0:
            # the state sits on the last sample of a piece: continue on the straightest successor (or end here)
            u = [j for j in self.succ[i] if not em[j][:3].any()]
            if u:
                i, ix = self.straightest(i, u), 0
                n_i, k_end = len(self.pts[i]), len(self.pts[i]) - 1
        # emitted samples ahead on the own piece (at least 1 m ahead)
        k_e = None
        for k in range(2, KMAX + 1):
            if ix + k < n_i and em[i][ix + k]:
                k_e = k
                break
        at_end = k_end <= KMAX
        ev, child, ptr_ref = CONT, None, None
        mark = None
        if k_e is not None and (not at_end or k_e < k_end):
            ev, k_use = MERGE, k_e
            q_star = self.pts[i][ix + k_e]
            ptr_ref = (i, ix + k_e)
            mark = (i, ix, ix + k_e - 1)
        elif at_end:
            k_use = max(k_end, 1)
            succ = self.succ[i]
            traced = [j for j in succ if em[j][:3].any()]
            untraced = [j for j in succ if not em[j][:3].any()]
            q_star = self.pts[i][-1]
            mark = (i, ix, n_i - 1)
            if not succ:
                ev = BORDER if self.open_end[i] else END
            elif not untraced:
                ev = MERGE
                j = self.straightest(i, traced)
                ptr_ref = (j, 0)
            elif len(untraced) >= 2:
                ev = FORK
                j = self.straightest(i, untraced)
                child = [x for x in untraced if x != j][0]
        else:
            k_use, q_star = 4, self.pts[i][ix + 4]
            for c in (6.0, 4.0, 2.0):
                kk = int(round(c / DS))
                seg = self.pts[i][ix : ix + kk + 1]
                a, b = seg[0], seg[-1]
                ch = b - a
                nrm = np.array([-ch[1], ch[0]]) / max(np.linalg.norm(ch), 1e-9)
                dev = float(np.abs((seg - a) @ nrm).max())
                if dev <= 0.3 or c == 2.0:
                    k_use, q_star = kk, b
                    break
            mark = (i, ix, ix + k_use)
        c_eff = max(k_use * DS, 0.5)
        chord = q_star - p
        ln = float(np.linalg.norm(chord))
        if ln < 0.3:
            chord = unit(theta) * 0.3
            ln = 0.3
        cdir = chord / ln
        phi = wrap(ang(chord) - theta)
        kb = int(np.clip(round(phi / BIN) + N_BIN // 2, 0, N_BIN - 1))
        res_phi = float(np.clip(phi - (kb - N_BIN // 2) * BIN, -math.radians(6.0), math.radians(6.0)) / math.radians(6.0))
        cc = 0
        while cc < 2 and c_eff > STEPS[cc] + 1e-6:
            cc += 1
        res_len = float(np.clip(ln - STEPS[cc], -2.0, 2.0))
        if ev == CONT and max(abs(q_star[0]), abs(q_star[1])) >= CLIP - 0.05:
            ev = BORDER
        nrm = np.array([-cdir[1], cdir[0]])
        qc = q_star + noise_lat * nrm + noise_lon * cdir
        out = dict(piece=i, ix=ix, dist=dist, k=kb, c=cc, res_phi=res_phi, res_len=res_len, ev=ev, q_star=q_star, qc=qc, off=-noise_lat, cdir=cdir, ptr_ref=ptr_ref, psi=0.0, child=None, mark=mark)
        if ev == FORK and child is not None:
            out["psi"] = float(wrap(ang(self.tan[child][0]) - ang(cdir)))
            out["child"] = child
        return out


def sample_check(gt, rng, n=200):
    """self test: expert steps from exact GT states (no noise) must stay on the lines and make sense"""
    G = ExpertGraph(gt, False)
    em = [np.zeros(len(p), bool) for p in G.pts]
    out = dict(n=0, dev=[], ev=np.zeros(5, int), c=np.zeros(3, int))
    if not G.has:
        return out
    for _ in range(n):
        j = rng.randint(len(G.allp))
        i, ix = int(G.own[j]), int(G.idx[j])
        if len(G.pts[i]) < 3:
            continue
        ix = min(ix, len(G.pts[i]) - 2)
        th = ang(G.tan[i][ix])
        t = G.targets(G.pts[i][ix], th, em)
        if t is None:
            continue
        out["n"] += 1
        d = G.tree.query(t["q_star"])[0]
        out["dev"].append(d)
        out["ev"][t["ev"]] += 1
        out["c"][t["c"]] += 1
    return out
