import sys
sys.path.insert(0, "../../evaluation")
import qgraph_readout as QR
s = QR.parse("mnms:1.75s0.2@0.5"); print(s, QR.name(s)); assert s.sigma == 0.2 and QR.name(s) == "mnms1.75s0.2_t0.5"
s2 = QR.parse("mnms:1.5@0.5"); assert s2.sigma is None and QR.name(s2) == "mnms1.5_t0.5"
print("ok")
