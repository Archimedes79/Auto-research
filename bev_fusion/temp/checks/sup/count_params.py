"""throw-away: parameter counts of the flow-block test model (best setup) and the simple timed model."""
import sys
import os
sys.path[:0] = ["tests", "training", "."]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.argv = ["x", "--sharp", "--pyr", "--doppler", "--data", "nuscenes", "--gate"]
import importlib
fe = importlib.import_module("test_flow_explains")
m = fe.Model()
n = lambda mod: sum(p.numel() for p in mod.parameters())
print("flow block total", n(m), {k: n(v) for k, v in m.named_children()})
