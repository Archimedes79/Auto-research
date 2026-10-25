"""Loop 10: near-to-far wavefront driver for the learned trace network (and its oracle: the ground-truth expert). Bookkeeping only (heads, traces, vertex cloud, rings, events) and
graph assembly (vertex graph -> pieces with end(i) = start(j), the format of graph_eval); the decisions come from a policy object:
  policy.decide(items, frames) -> list of dict(q, on, ev, psi, ptr, cls)   items = list of dict(b, pos, theta, flag, hist, node, trace)
Usage: imported; p10_replay.py runs the expert replay on the ground truth (test T2 / experiment E92)."""
import math

import numpy as np

import p10_expert as EX

RINGS = (12.5, 25.0, 37.5, 50.0, 70.0)
CAP = 6000


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Head:
    __slots__ = ("trace", "pos", "theta", "flag", "hist", "active", "parked", "miss", "pend", "node", "steps", "born")

    def __init__(self, trace, pos, theta, flag, node, born=0):
        self.trace, self.pos, self.theta, self.flag, self.node = trace, np.asarray(pos, np.float64), float(theta), int(flag), node
        self.hist = []  # last vertices (x, y, step class)
        self.active, self.parked, self.miss, self.pend, self.steps, self.born = True, False, 0, [], 0, born


class FrameState:
    def __init__(self):
        self.xy = []  # vertex coordinates (node id = index)
        self.traces = []  # list of dict(flag, nodes)
        self.edges = set()  # travel-order edges between nodes
        self.heads = []
        self.cloud = np.zeros((CAP, 6), np.float32)  # x, y, travel dx, travel dy, is_vertex, node
        self.nc = 0
        self.seeds = []  # pending seeds: (range, pos, theta, score)
        self.n_fork = self.n_merge = 0
        self.node_tok = {}  # node id -> (token h2, vertex position, cross-section positions) of the step that produced it (navigation head)

    # -------------------------------------------------------------- vertices
    def new_node(self, xy):
        self.xy.append(np.asarray(xy, np.float64))
        return len(self.xy) - 1

    def add_cloud(self, a, b, node_b, flag=1, step_ds=1.0):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return
        u = d / L
        n = max(int(math.ceil(L / step_ds)), 1)
        for k in range(1, n + 1):
            if self.nc >= CAP:
                return
            p = a + d * (k / n)
            self.cloud[self.nc] = (p[0], p[1], u[0] * flag, u[1] * flag, 1.0 if k == n else 0.0, node_b if k == n else -1)
            self.nc += 1

    def spawn(self, pos, theta, flag, node=None, born=0):
        if node is None:
            node = self.new_node(pos)
            if self.nc < CAP:
                self.cloud[self.nc] = (pos[0], pos[1], math.cos(theta) * flag, math.sin(theta) * flag, 1.0, node)
                self.nc += 1
        h = Head(len(self.traces), pos, theta, flag, node, born)
        self.traces.append(dict(flag=flag, nodes=[node]))
        self.heads.append(h)
        return h

    def emit(self, h, q, cls=0):
        node = self.new_node(q)
        self.add_cloud(h.pos if not h.hist else h.hist[-1][:2], q, node, h.flag)
        self.traces[h.trace]["nodes"].append(node)
        self._edge(h.node, node, h.flag)
        h.node = node
        h.hist.append((float(q[0]), float(q[1]), cls))
        h.hist = h.hist[-6:]
        return node

    def _edge(self, a, b, flag):
        if a != b:
            self.edges.add((a, b) if flag > 0 else (b, a))

    def nearest_vertex(self, q, rad=6.0):
        if self.nc == 0:
            return None
        c = self.cloud[: self.nc]
        m = c[:, 4] > 0.5
        if not m.any():
            return None
        d = np.linalg.norm(c[m, :2] - np.asarray(q)[None], axis=1)
        k = int(d.argmin())
        return int(c[m][k, 5]) if d[k] <= rad else None

    # -------------------------------------------------------------- assembly
    def assemble(self):
        n = len(self.xy)
        out = [[] for _ in range(n)]
        inn = [[] for _ in range(n)]
        for a, b in self.edges:
            out[a].append(b)
            inn[b].append(a)
        special = [len(inn[v]) != 1 or len(out[v]) != 1 for v in range(n)]
        pieces, starts, ends = [], {}, {}
        for v in range(n):
            if not special[v]:
                continue
            for m in out[v]:
                path = [v, m]
                guard = 0
                while not special[path[-1]] and guard < 10000:
                    path.append(out[path[-1]][0])
                    guard += 1
                if len(path) >= 2 and path[-1] != path[0]:
                    pieces.append(path)
        for k, p in enumerate(pieces):
            starts.setdefault(p[0], []).append(k)
            ends.setdefault(p[-1], []).append(k)
        edges = [(i, j) for v, ii in ends.items() for i in ii for j in starts.get(v, []) if i != j]
        pts = [np.array([self.xy[v] for v in p], np.float64) for p in pieces]
        return dict(pts=pts, edges=edges, nodes=pieces)


def run(policy, B, seeds, rings=RINGS, wave=1, max_iter=400, r_cov=1.2, r_head=1.2, gap_max=2, verbose=False, seed_dirs=(1, -1), sequential=True, order="near", four_heads=False):
    """seeds[b] = list of (pos, theta, score). Returns the list of FrameStates (assemble() gives the graphs).
    sequential: a new seed is only taken when the frame has no active head (the previous seed's trace is complete within the ring), which avoids duplicate traces of one lane."""
    fr = [FrameState() for _ in range(B)]
    for b in range(B):
        fr[b].seeds = sorted([(float(np.linalg.norm(p)), np.asarray(p, np.float64), float(t), float(s)) for p, t, s in seeds[b]], key=lambda x: x[0], reverse=(order == "far"))
    for si, R in enumerate(rings):
        for it in range(max_iter):
            for f in fr:
                if sequential and any(h.active and not h.parked for h in f.heads):
                    continue
                nsp = 0
                keep = []
                for sd in f.seeds:
                    r, p, th, sc = sd
                    if r >= R or nsp >= wave:
                        keep.append(sd)
                        continue
                    cov = False
                    ud = np.array([math.cos(th), math.sin(th)])
                    if f.nc:
                        c_ = f.cloud[: f.nc]
                        dd = np.linalg.norm(c_[:, :2] - p[None], axis=1)
                        cov = bool(((dd < r_cov) & ((c_[:, 2:4] @ ud) > 0.7)).any())
                    if not cov:
                        for h in f.heads:
                            if (h.active or h.parked) and np.linalg.norm(h.pos - p) < r_head and math.cos(h.theta) * ud[0] * h.flag + math.sin(h.theta) * ud[1] * h.flag > 0.7:
                                cov = True
                                break
                    if cov:
                        continue
                    nsp += 1
                    node = f.new_node(p)
                    if f.nc < CAP:
                        f.cloud[f.nc] = (p[0], p[1], math.cos(th), math.sin(th), 1.0, node)
                        f.nc += 1
                    combos = [(1, th), (-1, wrap(th + math.pi))] + ([(1, wrap(th + math.pi)), (-1, th)] if four_heads else [])
                    for fl, tt in combos:
                        f.spawn(p, tt, fl, node=node, born=si)
                f.seeds = keep
            items = [dict(b=b, h=h) for b, f in enumerate(fr) for h in f.heads if h.active and not h.parked]
            pending = any(sd[0] < R for f in fr for sd in f.seeds)
            if not items and not pending:
                break
            if items:
                its = [dict(b=it_["b"], pos=it_["h"].pos, theta=it_["h"].theta, flag=it_["h"].flag, hist=it_["h"].hist, node=it_["h"].node, trace=it_["h"].trace, steps=it_["h"].steps) for it_ in items]
                decs = policy.decide(its, fr)
                for it_, d in zip(items, decs):
                    apply_decision(fr[it_["b"]], it_["h"], d, R, si, gap_max, len(rings))
        for f in fr:
            for h in f.heads:
                h.parked = False
        if verbose:
            print(f"ring {R}: heads {sum(len(f.heads) for f in fr)}, vertices {sum(len(f.xy) for f in fr)}", flush=True)
    return fr


def apply_decision(f, h, d, R, si, gap_max, n_rings):
    q, ev, on = np.asarray(d["q"], np.float64), d["ev"], d["on"]
    cls = d.get("cls", 0)
    if not on and ev in (EX.CONT, EX.FORK):
        h.miss += 1
        h.pend.append((q, cls))
        if h.miss > gap_max:
            h.active = False
            h.pend = []
            return
        h.theta = math.atan2(q[1] - h.pos[1], q[0] - h.pos[0]) if np.linalg.norm(q - h.pos) > 1e-3 else h.theta
        h.pos = q
        return
    if not on:
        h.active = False
        h.pend = []
        return
    for qq, cc in h.pend:
        f.emit(h, qq, cc)
    h.pend, h.miss = [], 0
    if ev == EX.MERGE:
        tgt = d.get("ptr")
        if tgt is not None and tgt != h.node:
            f._edge(h.node, tgt, h.flag)
            f.n_merge += 1
        elif tgt is None:
            f.emit(h, q, cls)
        h.active = False
        return
    prev = h.pos if not h.hist else np.asarray(h.hist[-1][:2])
    node = f.emit(h, q, cls)
    if d.get("tok") is not None:
        f.node_tok[node] = (d["tok"], q, d["cp"])
    th_new = math.atan2(q[1] - prev[1], q[0] - prev[0]) if np.linalg.norm(q - prev) > 1e-3 else h.theta
    if ev == EX.FORK:
        ch = f.spawn(q, wrap(th_new + d.get("psi", 0.0)), h.flag, node=node, born=si)
        ch.hist = list(h.hist)
        f.n_fork += 1
    if ev in (EX.END, EX.BORDER):
        h.active = False
        return
    h.pos, h.theta = q, th_new
    h.steps += 1
    if np.linalg.norm(q) > R and si < n_rings - 1:
        h.parked = True


class ExpertPolicy:
    """oracle: the ground-truth expert decides from exact states; 'emitted' = ground-truth samples covered by an earlier step of any head (identity-based).  T2 / E92."""

    def __init__(self, gts):
        self.G = [(EX.ExpertGraph(g, False), EX.ExpertGraph(g, True)) for g in gts]
        self.mark = [[np.zeros(len(p), bool) for p in G[0].pts] for G in self.G]  # forward orientation

    def decide(self, items, fr):
        out = []
        for it in items:
            b, flag = it["b"], it["flag"]
            f = fr[b]
            G = self.G[b][0 if flag > 0 else 1]
            em = self.mark[b] if flag > 0 else [x[::-1] for x in self.mark[b]]
            t = G.targets(it["pos"], it["theta"], em)
            if t is None:
                out.append(dict(q=it["pos"], on=False, ev=EX.END))
                continue
            if t["mark"] is not None:
                pc, a, bb = t["mark"]
                n = len(G.pts[pc])
                if flag > 0:
                    self.mark[b][pc][a : bb + 1] = True
                else:
                    self.mark[b][pc][n - 1 - bb : n - a] = True
            ev = t["ev"]
            ptr = None
            if ev == EX.MERGE and t["ptr_ref"] is not None:
                ptr = f.nearest_vertex(t["q_star"], rad=4.0)
            if ev == EX.MERGE and ptr is None:
                ev = EX.CONT
            out.append(dict(q=t["q_star"], on=True, ev=ev, psi=t.get("psi", 0.0), ptr=ptr, cls=t["c"]))
        return out


def gt_seeds(gt, rng, per_piece=1):
    """oracle seeds: one point per piece (random sample), heading = travel direction"""
    import graph_eval as GE

    out = []
    for p in gt["pts"]:
        q = GE.resample(np.asarray(p, np.float64), 0.5)
        if len(q) < 3:
            continue
        for _ in range(per_piece):
            k = rng.randint(1, len(q) - 1)
            t = q[k + 1 if k + 1 < len(q) else k] - q[k - 1]
            out.append((q[k], math.atan2(t[1], t[0]), 1.0))
    return out


def orient_by_field(f, dirmap, min_votes=2):
    """OPTIONAL (E99c, not part of the learned main path): flip every traced piece whose mean agreement with the network's own direction head (dirmap [2, 200, 200], ego frame, travel direction) is negative, then re-assemble.
    Hard sign vote over a learned field, comparable to the orientation step of the rule pipeline."""
    g = f.assemble()
    flips = 0
    for path in g["nodes"]:
        votes = []
        for a, b in zip(path[:-1], path[1:]):
            pa, pb = f.xy[a], f.xy[b]
            d = pb - pa
            nrm = np.linalg.norm(d)
            if nrm < 1e-6:
                continue
            m = (pa + pb) / 2
            r = int(np.clip(np.floor((m[0] + 50.0) / 0.5), 0, 199))
            c = int(np.clip(np.floor((m[1] + 50.0) / 0.5), 0, 199))
            votes.append(float(dirmap[0, r, c] * d[0] + dirmap[1, r, c] * d[1]) / nrm)
        if len(votes) >= min_votes and np.mean(votes) < 0:
            flips += 1
            for a, b in zip(path[:-1], path[1:]):
                if (a, b) in f.edges:
                    f.edges.discard((a, b))
                    f.edges.add((b, a))
    return f.assemble(), flips
