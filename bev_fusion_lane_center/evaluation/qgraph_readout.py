"""Read-out steps for the nodes the lane head delivers (E275). Pure numpy, no model, no GPU.

    keep = suppress_close_nodes(pos, e, radius=1.5, thr=0.5)        # (M,) bool: the nodes that stay
    spec = parse("close:1.5@0.5")                                   # "plain@0.4" = the node threshold only; "close:<R>@<thr>" = close-node suppression, then the threshold
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

Spec = namedtuple("Spec", "kind radius thr")        # kind "plain" (radius None) or "close"
_SPEC = re.compile(r"^(plain|close:(\d+(?:\.\d+)?))@(\d+(?:\.\d+)?)$")


def parse(text):
    """'plain@0.5' -> Spec('plain', None, 0.5); 'close:1.5@0.4' -> Spec('close', 1.5, 0.4)"""
    m = _SPEC.match(text.strip())
    if not m:
        raise ValueError(f"read-out spec {text!r}: expected plain@<thr> or close:<radius>@<thr>")
    spec = Spec("plain" if m.group(1) == "plain" else "close", None if m.group(2) is None else float(m.group(2)), float(m.group(3)))
    if not spec.thr > 0.0:                                         # a dropped node has e = 0: with a threshold of 0 it would be delivered again
        raise ValueError(f"read-out spec {text!r}: the threshold must be above 0")
    return spec


def name(spec):
    """folder-safe name: plain_t0.5, close1.5_t0.4"""
    return f"plain_t{spec.thr:g}" if spec.kind == "plain" else f"close{spec.radius:g}_t{spec.thr:g}"


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


def apply(pred, spec):
    """the prediction as the read-out of `spec` sees it. plain: `pred` itself. close: a shallow copy whose nodes['e'] is 0 for every node that was not kept (the thresholded set is kept
    exactly as suppress_close_nodes returns it; the other arrays and the links are shared, not copied)"""
    if spec.kind == "plain":
        return pred
    nd = pred["nodes"]
    e = np.asarray(nd["e"], float)
    keep = suppress_close_nodes(nd["pos"], e, spec.radius, spec.thr)
    q = copy.copy(pred)
    q["nodes"] = dict(nd)
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
