"""scratch: the default (E273) node loss and link loss of the toy batch of test_tier_loss.py, to freeze them in the test (the critic pass confirmed the defaults equal the committed code)"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "training", "grid"))
import test_tier_loss as T  # noqa: E402

T.reset()
no, b = T.make(True)
l, info = T.QG.node_loss(no, b)
print("NODE", repr(float(l)), {k: round(v, 6) for k, v in info.items()})
import importlib.util
spec = importlib.util.spec_from_file_location("qgraph_head", os.path.join(HERE, "qgraph_head_copy.py"))
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
no, b = T.make(True)
lh, _ = H.node_loss(no, b)
print("HEAD", repr(float(lh)), "equal:", float(lh) == float(l))
