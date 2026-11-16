import sys, numpy as np
sys.path.insert(0, "../../evaluation")
import qgraph_readout as RO
s = RO.parse("mnms:1.5@0.5"); assert s.kind == "mnms" and s.radius == 1.5 and RO.name(s) == "mnms1.5_t0.5"
pos = np.array([[0, 0], [0.3, 0], [5, 5], [0.9, 0]], float); e = np.array([0.95, 0.9, 0.8, 0.7])
d = RO.matrix_nms(pos, e, 1.5)
print(d)
assert d[0] == 0.95 and d[2] == 0.8, d                    # the top node and the far node are untouched
assert d[1] < 0.9 and d[3] < 0.7, d                       # the close ones are decayed
q = RO.apply(dict(nodes=dict(pos=pos, e=e)), s)
print(q["nodes"]["e"])
print("ok")
