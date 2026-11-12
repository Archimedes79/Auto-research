p = "evaluation/qgraph_readout.py"
s = open(p, encoding="utf-8").read()
s = s.replace('Spec = namedtuple("Spec", "kind radius thr")        # kind "plain" (radius None) or "close"\n_SPEC = re.compile(r"^(plain|close:(\\d+(?:\\.\\d+)?))@(\\d+(?:\\.\\d+)?)$")',
              'Spec = namedtuple("Spec", "kind radius thr")        # kind "plain" (radius None), "close" or "mnms" (E288: one-pass Matrix-NMS decay)\n_SPEC = re.compile(r"^(plain|close:(\\d+(?:\\.\\d+)?)|mnms:(\\d+(?:\\.\\d+)?))@(\\d+(?:\\.\\d+)?)$")')
s = s.replace('''    m = _SPEC.match(text.strip())
    if not m:
        raise ValueError(f"read-out spec {text!r}: expected plain@<thr> or close:<radius>@<thr>")
    spec = Spec("plain" if m.group(1) == "plain" else "close", None if m.group(2) is None else float(m.group(2)), float(m.group(3)))''',
'''    m = _SPEC.match(text.strip())
    if not m:
        raise ValueError(f"read-out spec {text!r}: expected plain@<thr>, close:<radius>@<thr> or mnms:<radius>@<thr>")
    kind = "plain" if m.group(1) == "plain" else ("close" if m.group(2) is not None else "mnms")
    spec = Spec(kind, None if kind == "plain" else float(m.group(2) if kind == "close" else m.group(3)), float(m.group(4)))''')
s = s.replace('''    return f"plain_t{spec.thr:g}" if spec.kind == "plain" else f"close{spec.radius:g}_t{spec.thr:g}"''', '''    return f"plain_t{spec.thr:g}" if spec.kind == "plain" else f"{spec.kind}{spec.radius:g}_t{spec.thr:g}"''')
old = "def apply(pred, spec):"
new = '''def matrix_nms(pos, e, radius, thr=0.5, sigma=0.5):
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


def apply(pred, spec):'''
assert old in s
s = s.replace(old, new, 1)
old2 = '''    nd = pred["nodes"]
    e = np.asarray(nd["e"], float)
    keep = suppress_close_nodes(nd["pos"], e, spec.radius, spec.thr)
    q = copy.copy(pred)
    q["nodes"] = dict(nd)
    q["nodes"]["e"] = np.where(keep, e.reshape(-1), 0.0).reshape(e.shape)
    return q'''
new2 = '''    nd = pred["nodes"]
    e = np.asarray(nd["e"], float)
    q = copy.copy(pred)
    q["nodes"] = dict(nd)
    if spec.kind == "mnms":                                                  # E288: the decayed scores replace e (a node below the threshold after the decay is not delivered)
        q["nodes"]["e"] = matrix_nms(nd["pos"], e, spec.radius, spec.thr).reshape(e.shape)
        return q
    keep = suppress_close_nodes(nd["pos"], e, spec.radius, spec.thr)
    q["nodes"]["e"] = np.where(keep, e.reshape(-1), 0.0).reshape(e.shape)
    return q'''
assert old2 in s
s = s.replace(old2, new2, 1)
s = s.replace('    spec = parse("close:1.5@0.5")                                   # "plain@0.4" = the node threshold only; "close:<R>@<thr>" = close-node suppression, then the threshold',
              '    spec = parse("close:1.5@0.5")                                   # "plain@0.4" = the node threshold only; "close:<R>@<thr>" = close-node suppression, then the threshold; "mnms:<R>@<thr>" = Matrix-NMS (E288)')
open(p, "w", encoding="utf-8").write(s)
print("ok")
