"""Loop 18 / R2.3: numba kernels for the path union of the assembly (p16_infer.FastUnion), identical results.
The union is built node by node (about 330 nodes per frame); every step asks for the nearest vertex, the nearest edge and the vertices that lie on a new edge. FastUnion answers with numpy calls on tiny
arrays, so the time is Python / numpy call overhead. NumbaUnion keeps the same data (vertex array, sorted edge list, the adjacency sets and the path search stay Python) but answers the three questions with
compiled loops that perform the same floating-point operations in the same order, and the same tie-breaking (first index of the minimum in the sorted edge order). The sort of the vertices on a new
edge stays in numpy. Check: evaluation/drivers/p18_asm_check.py compares the graphs of fast=True and fast=2 on the validation frames.
Usage: p16_infer.assemble(..., fast=2)."""
import bisect
import math

import numpy as np
from numba import njit

from p16_infer import FastUnion


@njit(cache=True)
def _nearest_vertex(A, n, px, py):
    best, k = 1e300, -1
    for i in range(n):
        dx = A[i, 0] - px
        dy = A[i, 1] - py
        d = math.sqrt(dx * dx + dy * dy)
        if d < best:
            best, k = d, i
    return k, best


@njit(cache=True)
def _nearest_edge(A, E, ne, px, py):
    best, k = 1e300, -1
    bx, by = 0.0, 0.0
    for i in range(ne):
        a, b = E[i, 0], E[i, 1]
        ax, ay = A[a, 0], A[a, 1]
        abx = A[b, 0] - ax
        aby = A[b, 1] - ay
        den = abx * abx + aby * aby
        if den < 1e-12:
            den = 1e-12
        t = ((px - ax) * abx + (py - ay) * aby) / den
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        qx = ax + t * abx
        qy = ay + t * aby
        dx = qx - px
        dy = qy - py
        d = math.sqrt(dx * dx + dy * dy)
        if d < best:
            best, k, bx, by = d, i, qx, qy
    return k, best, bx, by


@njit(cache=True)
def _on_edge(A, n, pax, pay, pcx, pcy, a, c, snap, end):
    """vertices that lie on the new edge a -> c: (indices, t)"""
    abx = pcx - pax
    aby = pcy - pay
    den = abx * abx + aby * aby
    if den < 1e-12:
        den = 1e-12
    idx = np.empty(n, np.int64)
    tt = np.empty(n, np.float64)
    m = 0
    for i in range(n):
        if i == a or i == c:
            continue
        vx, vy = A[i, 0], A[i, 1]
        t = ((vx - pax) * abx + (vy - pay) * aby) / den
        if not (t > 0.0 and t < 1.0):
            continue
        tc = t
        qx = pax + tc * abx
        qy = pay + tc * aby
        dx = vx - qx
        dy = vy - qy
        if not (math.sqrt(dx * dx + dy * dy) < snap):
            continue
        dax = vx - pax
        day = vy - pay
        if not (math.sqrt(dax * dax + day * day) > end):
            continue
        dcx = vx - pcx
        dcy = vy - pcy
        if not (math.sqrt(dcx * dcx + dcy * dcy) > end):
            continue
        idx[m] = i
        tt[m] = t
        m += 1
    return idx[:m], tt[:m]


class NumbaUnion(FastUnion):
    def __init__(self):
        super().__init__()
        self._Ea = np.zeros((128, 2), np.int64)

    def nearest_vertex(self, p):
        if not self._n:
            return -1, 1e9
        k, d = _nearest_vertex(self._A, self._n, float(p[0]), float(p[1]))
        return int(k), float(d)

    def nearest_edge(self, p):
        ne = len(self._es)
        if not ne:
            return None, 1e9, None
        k, d, qx, qy = _nearest_edge(self._A, self._Ea, ne, float(p[0]), float(p[1]))
        return (int(self._Ea[k, 0]), int(self._Ea[k, 1])), float(d), np.array([qx, qy])

    def add_edge(self, a, b):
        if (a, b) not in self.E:
            pos = bisect.bisect_left(self._es, (a, b))
            ne = len(self._es)
            if ne == len(self._Ea):
                self._Ea = np.concatenate([self._Ea, np.zeros_like(self._Ea)])
            self._Ea[pos + 1 : ne + 1] = self._Ea[pos:ne].copy()
            self._Ea[pos] = (a, b)
            self._es.insert(pos, (a, b))
            self.E.add((a, b))
        self.adj.setdefault(a, set()).add(b)

    def split(self, e, proj):
        a, b = e
        w = self.add_vertex(proj)
        if (a, b) in self.E:
            self.E.discard((a, b))
            pos = bisect.bisect_left(self._es, (a, b))
            ne = len(self._es)
            self._Ea[pos : ne - 1] = self._Ea[pos + 1 : ne].copy()
            del self._es[pos]
        self.adj[a].discard(b)
        self.add_edge(a, w)
        self.add_edge(w, b)
        return w

    def add_split(self, a, c, snap=1.0, end=1.0):
        if a == c:
            return
        pa, pc = self.V[a], self.V[c]
        vs, tv = _on_edge(self._A, self._n, float(pa[0]), float(pa[1]), float(pc[0]), float(pc[1]), a, c, snap, end)
        vs = vs[np.argsort(tv)]
        chain = [a] + [int(v) for v in vs] + [c]
        for u, w in zip(chain[:-1], chain[1:]):
            if u != w and (u, w) not in self.E and not self.has_path(u, w):
                self.add_edge(u, w)
