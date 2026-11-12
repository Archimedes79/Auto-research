p = "evaluation/qgraph_readout.py"
s = open(p, encoding="utf-8").read()
s = s.replace('Spec = namedtuple("Spec", "kind radius thr sigma", defaults=(None,))', 'Spec = namedtuple("Spec", "kind radius thr sigma exempt", defaults=(None, False))')
s = s.replace('(?:s(\\d+(?:\\.\\d+)?))?)@(\\d+(?:\\.\\d+)?)$")', '(?:s(\\d+(?:\\.\\d+)?))?)(x)?@(\\d+(?:\\.\\d+)?)$")')
s = s.replace("float(m.group(5)), float(m.group(4)) if m.group(4) else None)", "float(m.group(6)), float(m.group(4)) if m.group(4) else None, bool(m.group(5)))")
s = s.replace("expected plain@<thr>, close:<radius>@<thr> or mnms:<radius>[s<sigma>]@<thr>", "expected plain@<thr>, close:<radius>[x]@<thr> or mnms:<radius>[s<sigma>][x]@<thr> (x = the nodes of the special types stop / exit / split / merge at p >= 0.5 are never suppressed)")
s = s.replace('''    return f"{spec.kind}{spec.radius:g}" + (f"s{spec.sigma:g}" if spec.sigma else "") + f"_t{spec.thr:g}"''', '''    return f"{spec.kind}{spec.radius:g}" + (f"s{spec.sigma:g}" if spec.sigma else "") + ("x" if spec.exempt else "") + f"_t{spec.thr:g}"''')
# suppress_close_nodes with protect
s = s.replace('def suppress_close_nodes(pos, e, radius, thr=0.5):\n    """-> keep (M,) bool; see the module docstring for the rule"""', 'def suppress_close_nodes(pos, e, radius, thr=0.5, protect=None):\n    """-> keep (M,) bool; see the module docstring for the rule. protect (M,) bool (E296): a candidate node that is protected is always kept (it still suppresses the weaker nodes around it)"""')
s = s.replace("    for i in order:\n        if n and np.min(((kept[:n] - pos[i]) ** 2).sum(1)) < r2:\n            continue", "    for i in order:\n        if n and np.min(((kept[:n] - pos[i]) ** 2).sum(1)) < r2 and not (protect is not None and protect[i]):\n            continue")
# matrix_nms with protect
s = s.replace("def matrix_nms(pos, e, radius, thr=0.5, sigma=0.5):", "def matrix_nms(pos, e, radius, thr=0.5, sigma=0.5, protect=None):")
s = s.replace("    out[order] = e[order] * decay\n    return np.where(out >= thr, out, 0.0)", "    if protect is not None:\n        decay = np.where(np.asarray(protect, bool)[order], 1.0, decay)                                      # E296: a protected node is not decayed (it still decays the weaker ones)\n    out[order] = e[order] * decay\n    return np.where(out >= thr, out, 0.0)")
# apply: build protect from the flags
old = "    nd = pred[\"nodes\"]\n    e = np.asarray(nd[\"e\"], float)\n    q = copy.copy(pred)\n    q[\"nodes\"] = dict(nd)\n    if spec.kind == \"mnms\":"
new = "    nd = pred[\"nodes\"]\n    e = np.asarray(nd[\"e\"], float)\n    q = copy.copy(pred)\n    q[\"nodes\"] = dict(nd)\n    protect = None\n    if spec.exempt:                                                          # E296: stop / exit / split / merge (flags 0 - 3 of `qtargets.FLAGS`) at p >= 0.5\n        protect = (np.asarray(nd[\"flags\"], float).reshape(len(e.reshape(-1)), -1)[:, :4] >= 0.5).any(1)\n    if spec.kind == \"mnms\":"
assert old in s
s = s.replace(old, new, 1)
s = s.replace("matrix_nms(nd[\"pos\"], e, spec.radius, spec.thr, spec.sigma or 0.5).reshape(e.shape)", "matrix_nms(nd[\"pos\"], e, spec.radius, spec.thr, spec.sigma or 0.5, protect).reshape(e.shape)")
s = s.replace("    keep = suppress_close_nodes(nd[\"pos\"], e, spec.radius, spec.thr)\n    q[\"nodes\"][\"e\"] = np.where(keep", "    keep = suppress_close_nodes(nd[\"pos\"], e, spec.radius, spec.thr, protect)\n    q[\"nodes\"][\"e\"] = np.where(keep")
open(p, "w", encoding="utf-8").write(s)
t = "temp/gt2/t_exempt.py"
open(t, "w", encoding="utf-8").write('''import sys
import numpy as np
sys.path.insert(0, "../../evaluation")
import qgraph_readout as QR
s = QR.parse("close:1.5x@0.5"); assert s.exempt and s.kind == "close" and QR.name(s) == "close1.5x_t0.5", s
s2 = QR.parse("mnms:1.75s0.1x@0.5"); assert s2.exempt and s2.sigma == 0.1 and QR.name(s2) == "mnms1.75s0.1x_t0.5"
assert not QR.parse("close:1.5@0.5").exempt
pos = np.array([[0, 0], [0.4, 0], [5, 5]], float); e = np.array([0.95, 0.9, 0.8])
flags = np.zeros((3, 8)); flags[1, 1] = 0.9                       # node 1 is an exit
pred = dict(nodes=dict(pos=pos, e=e, flags=flags))
assert QR.apply(pred, QR.parse("close:1.5@0.5"))["nodes"]["e"][1] == 0.0
assert QR.apply(pred, QR.parse("close:1.5x@0.5"))["nodes"]["e"][1] == 0.9
assert QR.apply(pred, QR.parse("mnms:1.75s0.1@0.5"))["nodes"]["e"][1] == 0.0
assert QR.apply(pred, QR.parse("mnms:1.75s0.1x@0.5"))["nodes"]["e"][1] == 0.9
print("ok")
''')
print("patched")
