import sys, json, os
sys.path.insert(0, "/mnt/c/Auto-research/autoresearch")
import numpy as np, torch
from evalsuite.check import _differs
from evalsuite.store import _identity_differences
from evalsuite.quantitative import same_question
from types import SimpleNamespace
print("torch tensors differ:", _differs({"x": torch.zeros(3)}, {"x": torch.ones(3) * 100}))
print("torch in list:", _differs([torch.zeros(3)], [torch.ones(3)]))
class P:  # an object with __slots__ / no __dict__
    __slots__ = ("v",)
    def __init__(self, v): self.v = v
print("slots obj:", _differs(P(np.zeros(2)), P(np.ones(2))))
print("np vs list:", _differs(np.zeros(2), [1.0, 1.0]))
B = "/mnt/c/Auto-research/bev_fusion/runs"
def st(r):
    m = json.load(open(f"{B}/{r}/eval/suite/quant/eval_n1000/manifest.json"))
    return SimpleNamespace(manifest=m, path=r)
for a, b in (("t8_ctl_s0", "t8_bevfeat_s0"), ("t2_ref_s0", "t8_ctl_s0"), ("t2_ref_s0", "t7_res4_s0")):
    try:
        same_question(st(a), st(b)); print(a, b, "accepted")
    except ValueError as e:
        print(a, b, "REFUSED:", str(e)[:300])
# resume tolerance: stored without gt_signature, new with another value
print(_identity_differences({"source": {"a": 1}}, {"source": {"a": 1, "gt_signature": "X"}}, ("source",), gained=[]))
print(_identity_differences({"source": {"a": 1, "gt_signature": "Y"}}, {"source": {"a": 1, "gt_signature": "X"}}, ("source",), gained=[]))
print(_identity_differences({"source": {"a": 1, "gt_signature": "Y"}}, {"source": {"a": 1}}, ("source",), gained=[]))
