"""The contract between the lane head (E273), the scorer and the pictures: ONE format for what the head delivers per frame (numpy, metres, the lane frame x right / y forward), and the
ORACLE prediction made from the lane GT (what a perfect network would deliver), so that scorer and pictures can be built and tested without a network.

Prediction of one frame (`pred`, a dict):
  nodes   dict   M nodes (the cells with a weight above the read-out threshold are the delivered ones)
            pos (M, 2) float, dirv (M, 2) unit direction, e (M,) existence probability, flags (M, 8) probabilities in the order of `qtargets.FLAGS`
            (stop, exit, split, merge, start, end, win_in, win_out), s (M,) section-start probability, cell (M, 2) int (row = x, column = y, 2 m cells),
            section (M,) int: the section (= delivered lane piece) of the node, the same number for the nodes of one piece (GT: the piece index; network: the argmax of the section membership)
  links   dict   L links between nodes: src (L,), dst (L,) node indices, p (L,) probability, long (L,) bool, a (L,), b (L,) Hermite handle scales (1/3 for short links),
            curves: None, or a list of L (K, 2) arrays that replace the Hermite curve (the oracle)
  assign  None or dict: P (M, K + 1) probabilities over the K map pieces and the dustbin (last column), map_ids (K,) the piece ids, road (M, R + 1) over roads + none, road_ids (R,)
  maps    None or dict of the dense layers for the pictures, PROBABILITIES (sigmoid applied, never logits): lines05 (6, 200, 200) = 3 line probabilities + 3 truncated distances in metres, centre1 (6, 100, 100)
            = centre probability, cos, sin, ox, oy, keep gate, events1 (6, 100, 100), road2 (2, 50, 50) = road, intersection
Read-out (output step, named; `evaluation/qgraph_eval.py`): nodes with e >= 0.5, links with p >= 0.5 between delivered nodes, one lane piece per section.

`oracle_prediction(g)` builds that dict from a lane GT graph: nodes and links of `qtargets.node_table`, probabilities 1, curves = `qtargets.decode`, assignment = each node to its own piece.
Run the self-test: python qformat.py
"""
import numpy as np

import qtargets as QT


def oracle_prediction(g, marker_dist=0.0, with_assign=True, sparse=None):
    nt = QT.node_table(g, marker_dist=marker_dist, sparse=sparse)                  # sparse = None: the E273 nodes; (eps, max_chord): the E276 sparse node target
    M, L = len(nt["pos"]), len(nt["src"])
    nodes = dict(pos=nt["pos"].copy(), dirv=nt["dirv"].copy(), e=np.ones(M), flags=nt["flags"].astype(float), s=nt["sstart"].astype(float), cell=nt["cell"].copy(),
                 section=nt["piece"].astype(int).copy())
    links = dict(src=nt["src"].copy(), dst=nt["dst"].copy(), p=np.ones(L), long=nt["long"].copy(), a=np.full(L, 1 / 3), b=np.full(L, 1 / 3), curves=QT.decode(nt))
    pred = dict(nodes=nodes, links=links, assign=None, maps=None, table=nt)
    if with_assign:
        ids = [p["piece"] for p in g["pieces"]]
        P = np.zeros((M, len(ids) + 1))
        P[np.arange(M), nt["piece"]] = 1.0
        roads = sorted({p["road"] for p in g["pieces"]})
        R = np.zeros((M, len(roads) + 1))
        for m in range(M):
            R[m, roads.index(g["pieces"][nt["piece"][m]]["road"])] = 1.0
        pred["assign"] = dict(P=P, map_ids=ids, road=R, road_ids=roads)
    return pred


def _selftest():
    import qtargets
    qtargets._selftest()
    ys = lambda a, b: np.stack([np.zeros(int((b - a) / 0.5) + 1), np.arange(a, b + 1e-9, 0.5)], 1)
    A = qtargets._piece("A", ys(-40, 40), start_open=True, end_open=True)
    ev = [dict(type="window", io="in", pos=[0.0, -40.0], **{"in": [], "out": ["A"]}, dirs_in=[], dirs_out=[[0, 1]], partial=False),
          dict(type="window", io="out", pos=[0.0, 40.0], **{"in": ["A"], "out": []}, dirs_in=[[0, 1]], dirs_out=[], partial=False)]
    g = dict(pieces=[A], events=ev, lines=dict(lane_divider=[], road_divider=[], boundary=[]))
    pr = oracle_prediction(g)
    n, l = pr["nodes"], pr["links"]
    assert len(n["pos"]) == len(n["section"]) == len(n["e"]) and len(l["src"]) == len(l["dst"]) == len(l["curves"]) == len(n["pos"]) - 1
    assert np.allclose(pr["assign"]["P"].sum(1), 1) and (n["section"] == 0).all()
    print("qformat selftest ok: nodes", len(n["pos"]), "links", len(l["src"]))


if __name__ == "__main__":
    _selftest()
