"""Lane geometry (numpy only): polylines in metres, resampling, tangents, simplification, and the small graph class every lane
metric and view of this package works on.

Coordinates are (u, v) in metres in any frame the adapter chooses (the plan view with v pointing up on the picture is the
natural one). A `LaneGraph` is a list of polylines (travel direction = point order when `directed`) and optional successor
`edges` (i, j): the end of polyline i continues into the start of polyline j. `nodes` (optional) are vertex ids per polyline
(the same id = the same vertex); they only serve the vertex count.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np


def length(p) -> float:
    p = np.asarray(p, np.float64)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def resample(p, ds: float) -> np.ndarray:
    """points every `ds` metres along the polyline (arc length), the last point always included"""
    p = np.asarray(p, np.float64)
    if len(p) < 2:
        return p.reshape(-1, 2).copy()
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 1e-9:
        return p[:1].copy()
    n = max(int(np.ceil(s[-1] / ds)), 1)
    t = np.linspace(0.0, s[-1], n + 1)
    return np.c_[np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])]


def tangents(p) -> np.ndarray:
    """unit tangent at every point (central differences, one-sided at the ends)"""
    p = np.asarray(p, np.float64)
    if len(p) < 2:
        return np.zeros((len(p), 2))
    d = np.gradient(p, axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.maximum(n, 1e-12)


def simplify(p, tol: float) -> np.ndarray:
    """indices kept by Douglas-Peucker with tolerance `tol` (iterative; first and last point always kept)"""
    p = np.asarray(p, np.float64)
    n = len(p)
    if n < 3:
        return np.arange(n)
    keep = np.zeros(n, bool)
    keep[[0, n - 1]] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        ab = p[b] - p[a]
        L = np.linalg.norm(ab)
        seg = p[a + 1:b] - p[a]
        d = np.linalg.norm(seg, axis=1) if L < 1e-12 else np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return np.flatnonzero(keep)


@dataclass
class Dense:
    """points sampled along a graph: xy (N, 2), line (N,) polyline index, s (N,) arc length from the start of the polyline,
    tan (N, 2) unit tangent, ds the spacing"""
    xy: np.ndarray
    line: np.ndarray
    s: np.ndarray
    tan: np.ndarray
    ds: float

    def __len__(self) -> int:
        return len(self.xy)


@dataclass
class LaneGraph:
    lines: List[np.ndarray] = field(default_factory=list)
    edges: List[Tuple[int, int]] = field(default_factory=list)
    directed: bool = True
    nodes: Optional[List[Any]] = None
    meta: dict = field(default_factory=dict)

    @staticmethod
    def of(obj) -> "LaneGraph":
        """accepts a LaneGraph, a dict(pts=[...], edges=[...], nodes=[...]) (the sparse-lane contract of the lane project) or a
        list of (n, 2) arrays"""
        if isinstance(obj, LaneGraph):
            return obj
        if isinstance(obj, np.ndarray) and obj.ndim == 2 and obj.shape[1] == 2:             # one polyline
            return LaneGraph([np.asarray(obj, np.float64)])
        if isinstance(obj, dict):
            pts = [np.asarray(p, np.float64).reshape(-1, 2) for p in obj.get("pts", obj.get("lines", []))]
            return LaneGraph(pts, [tuple(map(int, e)) for e in (obj.get("edges") or [])], bool(obj.get("directed", True)),
                             obj.get("nodes"), {k: v for k, v in obj.items() if k not in ("pts", "lines", "edges", "nodes", "directed")})
        return LaneGraph([np.asarray(p, np.float64).reshape(-1, 2) for p in obj])

    def __len__(self) -> int:
        return len(self.lines)

    def length(self) -> float:
        return float(sum(length(p) for p in self.lines))

    def dense(self, ds: float = 0.5, box: Optional[Tuple[float, float, float, float]] = None) -> Dense:
        """points every `ds` metres on every polyline; `box` = (u0, u1, v0, v1): only points inside are kept"""
        xy, ln, ss, tn = [], [], [], []
        for i, p in enumerate(self.lines):
            if len(p) < 2:
                continue
            q = resample(p, ds)
            seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
            xy.append(q)
            ln.append(np.full(len(q), i))
            ss.append(np.r_[0.0, np.cumsum(seg)])
            tn.append(tangents(q))
        if not xy:
            return Dense(np.zeros((0, 2)), np.zeros(0, int), np.zeros(0), np.zeros((0, 2)), ds)
        d = Dense(np.concatenate(xy), np.concatenate(ln), np.concatenate(ss), np.concatenate(tn), ds)
        if box is not None:
            u0, u1, v0, v1 = box
            k = (d.xy[:, 0] >= u0) & (d.xy[:, 0] <= u1) & (d.xy[:, 1] >= v0) & (d.xy[:, 1] <= v1)
            d = Dense(d.xy[k], d.line[k], d.s[k], d.tan[k], ds)
        return d

    def vertices(self, tol: float = 0.3) -> int:
        """number of vertices: the unique ids of `nodes` when the graph carries them, else the Douglas-Peucker vertices of every
        polyline at `tol` (the definition that makes graphs without bookkeeping comparable)"""
        if self.nodes is not None and len(self.nodes) == len(self.lines):
            ids = set()
            for k, nd in enumerate(self.nodes):
                ids.update((k, j) for j in range(len(self.lines[k]))) if nd is None else ids.update(int(i) for i in nd)
            return len(ids)
        return int(sum(len(simplify(p, tol)) for p in self.lines if len(p) >= 2))

    def turning_deg(self, ds: float = 1.0) -> float:
        """total absolute change of heading along all polylines in degrees (curviness)"""
        tot = 0.0
        for p in self.lines:
            q = resample(p, ds)
            if len(q) < 3:
                continue
            d = np.diff(q, axis=0)
            a = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
            tot += float(np.abs(np.diff(a)).sum())
        return float(np.degrees(tot))

    def point_graph(self, ds: float = 1.0, link: float = 0.0):
        """(xy (N, 2), edges (M, 2) int) of the graph resampled every `ds`: consecutive points of a polyline are linked (both ways if
        not directed), end -> start of `edges`, and, when `link` > 0, ends / starts closer than `link` metres (the undirected
        skeleton case: pieces meeting at a junction); identical coordinates are one node"""
        xy, starts, ends, ed = [], [], [], []
        n0 = 0
        for p in self.lines:
            q = resample(p, ds) if len(p) >= 2 else np.asarray(p, np.float64).reshape(-1, 2)
            if not len(q):
                starts.append(-1); ends.append(-1)
                continue
            xy.append(q)
            idx = np.arange(n0, n0 + len(q))
            ed += list(zip(idx[:-1], idx[1:]))
            starts.append(n0); ends.append(n0 + len(q) - 1)
            n0 += len(q)
        if not xy:
            return np.zeros((0, 2)), np.zeros((0, 2), int)
        xy = np.concatenate(xy)
        for i, j in self.edges:
            if 0 <= i < len(ends) and 0 <= j < len(starts) and ends[i] >= 0 and starts[j] >= 0:
                ed.append((ends[i], starts[j]))
        ed = np.asarray(ed, int).reshape(-1, 2)
        # merge identical coordinates (junction centroids shared by pieces)
        key = np.round(xy, 4)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        xy, ed = xy[first], inv[ed] if len(ed) else ed
        if link > 0:
            from scipy.spatial import cKDTree
            ends_all = np.array([inv[e] for e in ends + starts if e >= 0], int)
            t = cKDTree(xy[ends_all]) if len(ends_all) else None
            if t is not None:
                for a, b in t.query_pairs(link):
                    ed = np.vstack([ed, [ends_all[a], ends_all[b]]])
        ed = ed[ed[:, 0] != ed[:, 1]] if len(ed) else ed
        if not self.directed and len(ed):
            ed = np.vstack([ed, ed[:, ::-1]])
        return xy, ed
