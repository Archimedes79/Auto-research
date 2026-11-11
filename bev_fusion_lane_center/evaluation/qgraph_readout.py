"""Read-out steps for the nodes the lane head delivers (E275). Pure numpy, no model, no GPU.

    keep = suppress_close_nodes(pos, e, radius=1.5, thr=0.5)        # (M,) bool: the nodes that stay
    spec = parse("close:1.5@0.5")                                   # "plain@0.4" = the node threshold only; "close:<R>@<thr>" = close-node suppression, then the threshold; "mnms:<R>@<thr>" = Matrix-NMS (E288)
    q = apply(pred, spec)                                           # the prediction with the dropped nodes at e = 0: every consumer of the read-out (qgraph_eval.kept / node_counts /
                                                                    # assign_counts) then sees them as not delivered; plain returns `pred` itself

Close-node suppression (the named OUTPUT step of E275; no loss goes through it, so it is a hard selection on the output in the sense of D55 and not part of the differentiable path):
among the nodes with e >= thr, go through them by e descending (ties: the lower index first) and keep a node unless a node that is already kept lies STRICTLY closer than `radius`
metres (Euclidean, BEV metres; at exactly `radius` both stay: "closer than", not "within"). Ties: nodes with exactly equal e (a saturated sigmoid gives 1.0 more than once) are taken in index
order (raster order of the cells), so for exact ties the kept set depends on the order of the input; `tied_pairs` counts them. The result is a subset of the nodes with e >= thr in which no two nodes are closer than `radius`. A node below the
threshold is never kept and never suppresses (it could only suppress nodes of lower e, which are below the threshold too: equal to a rule that looks at the nodes from e >= 0.2 on, as the E274
scratch sweep did). The 3 x 3 cell variant of the E274 scratch sweep was an indexing error and is not part of this module.
Tests: evaluation/tests/test_qgraph_readout.py. Time: O(n * kept) per frame, numpy.
"""
import copy
import re
import time
from collections import namedtuple

import numpy as np

Spec = namedtuple("Spec", "kind radius thr sigma", defaults=(None,))        # kind "plain" (radius None), "close" or "mnms" (E288: one-pass Matrix-NMS decay, sigma = its Gaussian width, None = 0.5)
_SPEC = re.compile(r"^(plain|close:(\d+(?:\.\d+)?)|mnms:(\d+(?:\.\d+)?)(?:s(\d+(?:\.\d+)?))?)@(\d+(?:\.\d+)?)$")


def parse(text):
    """'plain@0.5' -> Spec('plain', None, 0.5); 'close:1.5@0.4' -> Spec('close', 1.5, 0.4)"""
    m = _SPEC.match(text.strip())
    if not m:
        raise ValueError(f"read-out spec {text!r}: expected plain@<thr>, close:<radius>@<thr> or mnms:<radius>[s<sigma>]@<thr>")
    kind = "plain" if m.group(1) == "plain" else ("close" if m.group(2) is not None else "mnms")
    spec = Spec(kind, None if kind == "plain" else float(m.group(2) if kind == "close" else m.group(3)), float(m.group(5)), float(m.group(4)) if m.group(4) else None)
    if not spec.thr > 0.0:                                         # a dropped node has e = 0: with a threshold of 0 it would be delivered again
        raise ValueError(f"read-out spec {text!r}: the threshold must be above 0")
    return spec


def name(spec):
    """folder-safe name: plain_t0.5, close1.5_t0.4"""
    if spec.kind == "plain":
        return f"plain_t{spec.thr:g}"
    return f"{spec.kind}{spec.radius:g}" + (f"s{spec.sigma:g}" if spec.sigma else "") + f"_t{spec.thr:g}"


def suppress_close_nodes(pos, e, radius, thr=0.5):
    """-> keep (M,) bool; see the module docstring for the rule"""
    pos = np.asarray(pos, float).reshape(-1, 2)
    e = np.asarray(e, float).reshape(-1)
    assert len(pos) == len(e), (pos.shape, e.shape)
    keep = np.zeros(len(e), bool)
    cand = np.flatnonzero(e >= thr)
    assert np.isfinite(pos[cand]).all(), "a candidate node has a non-finite position"      # a NaN would stop every later suppression of the frame
    order = cand[np.argsort(-e[cand], kind="stable")]               # e descending, equal e: the lower index first
    kept = np.empty((len(order), 2))
    n = 0
    r2 = float(radius) ** 2
    for i in order:
        if n and np.min(((kept[:n] - pos[i]) ** 2).sum(1)) < r2:
            continue
        kept[n] = pos[i]
        n += 1
        keep[i] = True
    return keep


def matrix_nms(pos, e, radius, thr=0.5, sigma=0.5):
    """E288: the one-pass Matrix-NMS decay (SOLOv2, Wang et al. 2020) on the nodes with e >= thr: the overlap of two nodes is o = max(0, 1 - d / radius); going through the nodes by e descending,
    the decay of node j is the minimum over the higher nodes i of exp(-(o_ij^2 - c_i^2) / sigma) with the compensation c_i = max over the nodes k above i of o_ki (a node that is itself suppressed
    suppresses less); the decayed score is e_j x decay_j. -> (M,) float: the decayed scores (0 below thr). No sequential keep set: one matrix product-free pass, differentiable in e and pos"""
    pos = np.asarray(pos, float).reshape(-1, 2)
    e = np.asarray(e, float).reshape(-1)
    out = np.zeros(len(e))
    cand = np.flatnonzero(e >= thr)
    if not len(cand):
        return out
    order = cand[np.argsort(-e[cand], kind="stable")]
    P = pos[order]
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    o = np.triu(np.maximum(0.0, 1.0 - d / radius), 1)                        # o[i, j], i above j
    comp = o.max(0)                                                         # c_i = max_k<i o[k, i] (column i holds the overlaps of i with the nodes above it)
    decay = np.exp(-(o ** 2 - comp[:, None] ** 2) / sigma)
    decay = np.where(np.triu(np.ones_like(o), 1) > 0, decay, np.inf).min(0)   # over the higher nodes i < j
    decay = np.where(np.isfinite(decay), decay, 1.0)
    out[order] = e[order] * decay
    return np.where(out >= thr, out, 0.0)


def apply(pred, spec):
    """the prediction as the read-out of `spec` sees it. plain: `pred` itself. close: a shallow copy whose nodes['e'] is 0 for every node that was not kept (the thresholded set is kept
    exactly as suppress_close_nodes returns it; the other arrays and the links are shared, not copied)"""
    if spec.kind == "plain":
        return pred
    nd = pred["nodes"]
    e = np.asarray(nd["e"], float)
    q = copy.copy(pred)
    q["nodes"] = dict(nd)
    if spec.kind == "mnms":                                                  # E288: the decayed scores replace e (a node below the threshold after the decay is not delivered)
        q["nodes"]["e"] = matrix_nms(nd["pos"], e, spec.radius, spec.thr, spec.sigma or 0.5).reshape(e.shape)
        return q
    keep = suppress_close_nodes(nd["pos"], e, spec.radius, spec.thr)
    q["nodes"]["e"] = np.where(keep, e.reshape(-1), 0.0).reshape(e.shape)
    return q


def counts(preds, spec):
    """(nodes delivered before, nodes delivered after) over the frames: the kept nodes at e >= thr, without and with the step"""
    before = sum(int((np.asarray(p["nodes"]["e"], float) >= spec.thr).sum()) for p in preds)
    after = sum(int((np.asarray(apply(p, spec)["nodes"]["e"], float) >= spec.thr).sum()) for p in preds)
    return before, after


def tied_pairs(preds, spec):
    """number of pairs of candidates (e >= thr) of one frame with exactly equal e that lie closer than the radius: the only place where the kept set depends on the order of the input"""
    n = 0
    for p in preds:
        e = np.asarray(p["nodes"]["e"], float).reshape(-1)
        pos = np.asarray(p["nodes"]["pos"], float).reshape(-1, 2)
        c = np.flatnonzero(e >= spec.thr)
        for v in np.unique(e[c]):
            g = c[e[c] == v]
            if len(g) > 1:
                d2 = ((pos[g][:, None, :] - pos[g][None, :, :]) ** 2).sum(-1)
                n += int(((d2 < spec.radius ** 2).sum() - len(g)) // 2)
    return n


def timed(preds, spec):
    """ms per frame of the step (numpy, one CPU thread, the frames one after the other)"""
    t0 = time.perf_counter()
    for p in preds:
        apply(p, spec)
    return 1000.0 * (time.perf_counter() - t0) / max(len(preds), 1)


def dump_full(preds, rows, path, e_min=0.2, p_min=0.05):
    """E276 (the analyses of the video notes: assignment, temporal consistency, endpoints, link uncertainty): the candidate nodes (e >= e_min) of every frame with ALL their numbers, the links between
    them (p >= p_min, node indices renumbered to the stored nodes), the assignment rows of those nodes (float32) and the two pixel layers the pictures need (road2 = road / intersection, events1, float16);
    [{'row', 'idx', 'e', 'pos', 'dirv', 'flags', 's', 'section', 'link_src', 'link_dst', 'link_p', 'link_long', 'link_a', 'link_b', 'assign_P', 'map_ids', 'assign_road', 'road_ids', 'road2', 'events1'}],
    pickled to `path` (scratch, about 150 KB per frame)"""
    import pickle
    out = []
    for p, r in zip(preds, rows):
        nd, lk = p["nodes"], p["links"]
        e = np.asarray(nd["e"], float).reshape(-1)
        k = np.flatnonzero(e >= e_min)
        new = -np.ones(len(e), int)
        new[k] = np.arange(len(k))
        src, dst = np.asarray(lk["src"]).reshape(-1).astype(int), np.asarray(lk["dst"]).reshape(-1).astype(int)
        pl = np.asarray(lk["p"], float).reshape(-1)
        m = (pl >= p_min) & (new[src] >= 0) & (new[dst] >= 0) & (src != dst)
        d = dict(row=int(r), idx=k, e=e[k], pos=np.asarray(nd["pos"], float).reshape(-1, 2)[k], dirv=np.asarray(nd["dirv"], float).reshape(-1, 2)[k],
                 flags=np.asarray(nd["flags"], float).reshape(len(e), -1)[k], s=np.asarray(nd["s"], float).reshape(-1)[k] if "s" in nd else None, section=np.asarray(nd["section"]).reshape(-1)[k],
                 link_src=new[src[m]], link_dst=new[dst[m]], link_p=pl[m], link_long=np.asarray(lk["long"]).reshape(-1).astype(bool)[m], link_a=np.asarray(lk["a"], float).reshape(-1)[m],
                 link_b=np.asarray(lk["b"], float).reshape(-1)[m])
        a = p.get("assign")
        if a is not None:
            d.update(assign_P=np.asarray(a["P"], np.float32)[k], map_ids=list(a["map_ids"]), assign_road=np.asarray(a["road"], np.float32)[k] if a.get("road") is not None else None,
                     road_ids=list(a.get("road_ids", [])))
        mp = p.get("maps")
        if mp is not None:
            d.update(road2=np.asarray(mp["road2"], np.float16), events1=np.asarray(mp["events1"], np.float16))
        out.append(d)
    with open(path, "wb") as f:
        pickle.dump(out, f, protocol=4)
    return len(out)


def dump_nodes(preds, rows, path, e_min=0.2):
    """the delivered-candidate nodes (e >= e_min) of every frame for the CPU analysis, values as the scorer sees them (float64 copies of the model's float32): [{'row', 'pos', 'dirv', 'e', 'flags', 'cell', 'section'}], pickled to `path` (scratch)"""
    import pickle
    out = []
    for p, r in zip(preds, rows):
        nd = p["nodes"]
        e = np.asarray(nd["e"], float).reshape(-1)
        k = np.flatnonzero(e >= e_min)
        out.append(dict(row=int(r), idx=k, e=e[k], pos=np.asarray(nd["pos"], float).reshape(-1, 2)[k], dirv=np.asarray(nd["dirv"], float).reshape(-1, 2)[k],
                        flags=np.asarray(nd["flags"], float).reshape(len(e), -1)[k], cell=np.asarray(nd["cell"]).reshape(len(e), -1)[k], section=np.asarray(nd["section"]).reshape(-1)[k]))
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return len(out)
