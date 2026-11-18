"""The delivered lane graph of one frame and its JSON form.

    Graph.lines   [(n_i, 2) float] polylines in metres (x right, y forward, ego at the origin), in driving direction
    Graph.edges   [(i, j)]        piece i continues into piece j (the end of i is the start of j)
    Graph.nodes   [[int]]         the node ids along each piece (B's active-node indices; connectors 10000 + n, 20000 + n)
    Graph.kinds   [str]           "lane" (a section of B), "link" (a link of B between sections), "connector" (the junction head)
"""
import json
from dataclasses import dataclass

import numpy as np


def simplify(P: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker (the last named output step): the vertices of the polyline P (n, 2) that keep it within tol metres of the point-to-SEGMENT
    distance (the project's `sparse_metrics._dp_keep`, which the vertex measure vert100_dp uses)"""
    keep = np.zeros(len(P), bool)
    keep[[0, -1]] = True
    stack = [(0, len(P) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        ab = P[b] - P[a]
        t = np.clip(((P[a + 1:b] - P[a]) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        d = np.linalg.norm(P[a + 1:b] - (P[a] + t[:, None] * ab), axis=1)
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return P[keep]


@dataclass
class Graph:
    lines: list
    edges: list
    nodes: list
    kinds: list

    def raw(self) -> dict:
        """the graph dict of the project's scorer (`evaluation/qgraph_eval.graph_counts`)"""
        return dict(pts=self.lines, edges=self.edges, nodes=self.nodes)

    def to_json(self, tol: float) -> dict:
        """the planner-facing form: polylines thinned to `tol` metres, 2 decimals"""
        pieces = [dict(kind=k, points=np.round(simplify(np.asarray(P, float), tol), 2).tolist()) for P, k in zip(self.lines, self.kinds)]
        return dict(pieces=pieces, edges=[[int(a), int(b)] for a, b in self.edges])


def write_json(path, graphs: dict, tol: float) -> None:
    """{frame token: Graph} -> one JSON file"""
    with open(path, "w") as f:
        json.dump({str(k): g.to_json(tol) for k, g in graphs.items()}, f)
