import sys
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
sp = QR.parse("close:1.5x0.95@0.5"); assert sp.exempt == 0.95 and QR.name(sp) == "close1.5x0.95_t0.5", sp
assert QR.apply(pred, sp)["nodes"]["e"][1] == 0.0          # the exit flag is 0.9 < 0.95: not protected
assert QR.parse("close:1.5x@0.5").exempt == 0.5 and QR.name(QR.parse("close:1.5x@0.5")) == "close1.5x_t0.5"
print("ok")
