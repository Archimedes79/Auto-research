"""Equivalence with the ORIGINAL modules (net/, evaluation) on stored VAL frames, CPU, float32, EXACT (no tolerance): every stage's outputs, the
delivered graphs of both read-outs (pieces, points, edges, node ids, kinds), the Douglas-Peucker path against the project's `_dp_keep`, golden digests
of three delivered graphs, and a fuzz of the read-out alone against the original read-out on synthetic frames.
The original runs at batch 4 (as B's stored dump): bit identity holds while no batch of 4 exceeds the original's 80,000 link candidates (above it the
original subsamples at random)."""
import dataclasses
import hashlib
import json

import numpy as np
import torch

from lg_config import CHAMPION
from lg_features import FeatureStore, split_rows
from lg_graph import simplify
from lg_model import LaneGraphNet, per_frame
from lg_project import original_graph, original_models
from lg_readout import deliver

# the champion's snapped graphs of the VAL frames 0, 400, 800 (index into split_rows("val")), seed 0, CPU float32: sha256 (16 hex) of the structure
# (kinds, edges, node ids, points per piece) and the sum of |coordinates| (m). The structure is exact; the sum is compared within GOLDEN_TOL because
# the CPU's float32 sums depend on the thread count in the last bits.
GOLDEN = {0: ("b6aad7495f9b35c5", 106702.358), 400: ("f995a6698c7a7133", 61283.089), 800: ("5c5d50234b4f26c2", 82095.247)}
GOLDEN_TOL = 1e-2


def digest(g) -> tuple:
    body = json.dumps([g.kinds, sorted(map(list, g.edges)), g.nodes, [len(P) for P in g.lines]])
    return hashlib.sha256(body.encode()).hexdigest()[:16], round(float(sum(np.abs(P).sum() for P in g.lines)), 3)


def golden_ok(got: dict) -> bool:
    return all(GOLDEN[k] is not None and got[k][0] == GOLDEN[k][0] and abs(got[k][1] - GOLDEN[k][1]) <= GOLDEN_TOL for k in GOLDEN)


def same_graph(raw: dict, kinds: list, g) -> bool:
    return (len(raw["pts"]) == len(g.lines) and all(np.array_equal(np.asarray(p), q) for p, q in zip(raw["pts"], g.lines))
            and sorted(map(tuple, raw["edges"])) == sorted(map(tuple, g.edges)) and [list(n) for n in raw["nodes"]] == g.nodes and kinds == g.kinds)


def original_pred(lanes: dict) -> dict:
    """a LaneFrame in the original prediction contract (qformat: nodes / links)"""
    sec = np.where(lanes["valid"][lanes["start"]], lanes["start"], np.arange(len(lanes["e"])))
    return dict(nodes=dict(pos=lanes["pos"], dirv=lanes["dir"], e=lanes["e"], section=sec, flags=np.zeros((len(sec), 8))),
                links=dict(src=lanes["src"], dst=lanes["dst"], p=lanes["p"], long=lanes["long"], a=lanes["sa"], b=lanes["sb"], curves=None))


def stages(seed: int, rows: list, batch: int = 4) -> dict:
    """-> {quantity: number of frames / batches where the champion differs from the original}, {piece kind: count compared}, {row: digest}"""
    torch.set_grad_enabled(False)
    qg, jm = original_models(seed)
    import qpredict as QP
    import sparse_metrics as SM
    net = LaneGraphNet.from_runs(seed, dataclasses.replace(CHAMPION, attached_ends=True))
    store = FeatureStore()
    bad, count = {}, {}

    def check(name, a, b):
        bad[name] = bad.get(name, 0) + (not (a.shape == b.shape and torch.equal(a, b)))
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        x = store(chunk)
        ob = qg(dict(x=x), upto="sections", frozen=("pyr", "nodes", "links", "sections"), gt=False)
        K = ob["L"]["valid"].shape[1]
        fake = dict(P_piece=torch.zeros(len(chunk), K, 1), P_road=torch.zeros(len(chunk), K, 1))                     # B's map assignment is not part of the graph
        preds = QP.predict_from_out(dict(ob, A=fake), dict(map_ids=[[]] * len(chunk), map_roads=[[]] * len(chunk), token=[""] * len(chunk)))
        _, jo, oe = jm(x)
        P = net.pyramid(x)
        for k in P:
            check(f"pyramid.{k}", P[k], ob["P"][k])
        no = net.nodes(P)
        for k in ("e", "flags", "dir_raw", "s", "uout", "uin", "tok", "pos"):
            check(f"nodes.{k}", no[k], ob["no"][k])
        L = net.links(no, P)
        for mine, theirs in (("act", "act"), ("valid", "valid"), ("pos", "pos"), ("dir", "dirv"), ("frame", "pb"), ("src", "pa"), ("dst", "pc"), ("long", "is_long"),
                             ("logit", "logit"), ("sa", "sa"), ("sb", "sb")):
            check(f"links.{mine}", L[mine], ob["L"][theirs])
        check("links.start", L["start"], ob["M"].argmax(-1))
        M = net.maps(P)
        for k in M:
            check(f"maps.{k}", M[k], jo[k])
        Q = net(x)["pairs"]
        for k in ("kp_pos", "kp_dir", "kp_p", "valid", "logit", "allowed", "scale", "att_pos", "att_dir", "att_p"):
            check(f"pairs.{k}", Q[k], oe[k])
        for b, (lanes, pairs) in enumerate(per_frame(dict(lanes=L, pairs=Q))):
            e = {k: oe[k][b].float().numpy() for k in ("kp_pos", "kp_dir", "logit", "scale", "att_pos", "att_dir")}
            e["allowed"] = oe["allowed"][b].numpy()
            for attached in (False, True):
                g = deliver(lanes, pairs, dataclasses.replace(CHAMPION, attached_ends=attached))
                name = "graph.attached" if attached else "graph.snapped"
                bad[name] = bad.get(name, 0) + (not same_graph(*original_graph(preds[b], e, attached), g))
                bad["graph.douglas_peucker"] = bad.get("graph.douglas_peucker", 0) + sum(not np.array_equal(simplify(P_, 0.3), P_[SM._dp_keep(P_, 0.3)]) for P_ in g.lines)
                for k in g.kinds:
                    count[k] = count.get(k, 0) + (not attached)
    return bad, count


def golden(seed: int = 0) -> dict:
    """{VAL index: digest} of the champion's snapped graphs of the GOLDEN frames"""
    rows = split_rows("val")
    net = LaneGraphNet.from_runs(seed)
    return {i: digest(g) for i, g in zip(GOLDEN, net.graphs(FeatureStore()([rows[i] for i in GOLDEN])))}


def fuzz(n: int = 300, seed: int = 0) -> int:
    """synthetic frames (random lanes, sections, links, long links, keypoint pairs near the piece ends): -> number where deliver differs from the original read-out"""
    rng = np.random.RandomState(seed)
    diff = 0
    for _ in range(n):
        pos, dirs, start, src, dst = [], [], [], [], []
        for lane in range(rng.randint(1, 6)):
            p0, h = rng.uniform(-40, 40, 2), rng.uniform(0, 2 * np.pi)
            t = np.array([np.cos(h), np.sin(h)])
            n0, m = len(pos), rng.randint(2, 8)
            cut = rng.randint(1, m + 1)                                                          # a section break inside the lane, or none
            for k in range(m):
                pos.append(p0 + 2.0 * k * t + rng.normal(0, 0.3, 2))
                dirs.append(t)
                start.append(n0 if k < cut else n0 + cut)
                if k:
                    src.append(n0 + k - 1), dst.append(n0 + k)
        K = len(pos)
        for _ in range(rng.randint(0, 4)):                                                      # links between lanes (long ones among them)
            a, b = rng.randint(K), rng.randint(K)
            if a != b and (a, b) not in zip(src, dst):
                src.append(a), dst.append(b)
        pos, dirs = np.array(pos), np.array(dirs)
        Lk = len(src)
        long = np.linalg.norm(pos[dst] - pos[src], axis=1) > 8.0
        lanes = dict(pos=pos, dir=dirs, e=rng.uniform(0.3, 1.0, K), valid=np.ones(K, bool), start=np.array(start), src=np.array(src), dst=np.array(dst),
                     p=rng.uniform(0.2, 1.0, Lk), sa=np.where(long, rng.uniform(0.1, 0.8, Lk), 1 / 3), sb=np.where(long, rng.uniform(0.1, 0.8, Lk), 1 / 3), long=long)
        J = 4
        ends = pos[rng.randint(K, size=2 * J)] + rng.normal(0, 2.0, (2 * J, 2))
        ang = rng.uniform(0, 2 * np.pi, 2 * J)
        pairs = dict(kp_pos=ends.astype(np.float32), kp_dir=np.c_[np.cos(ang), np.sin(ang)].astype(np.float32), logit=rng.normal(0, 3, (J, J)).astype(np.float32),
                     allowed=rng.uniform(size=(J, J)) < 0.7, scale=rng.uniform(0.05, 1.25, (J, J, 2)).astype(np.float32))
        pairs["att_pos"], pairs["att_dir"] = pairs["kp_pos"] + rng.normal(0, 1, (2 * J, 2)).astype(np.float32), pairs["kp_dir"]
        for attached in (False, True):
            diff += not same_graph(*original_graph(original_pred(lanes), pairs, attached), deliver(lanes, pairs, dataclasses.replace(CHAMPION, attached_ends=attached)))
    return diff
