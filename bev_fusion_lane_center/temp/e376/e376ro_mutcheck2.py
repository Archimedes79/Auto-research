"""throw-away: the zero-mass guard mutant against the stub-head expectation of test_anatomy (the first mutation run killed it only through the test's own source-text check)"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = sys.argv[:1]
import e376ro_test_jtoken_eval as T  # noqa: E402

E = T.E
out, _ = T.stub_out(7, {0: {1: 0.5, 2: 0.35}, 3: {4: 0.6}}, [0.1] * 7)
y = torch.zeros(1, 7, 7, dtype=torch.uint8)
for i, j in ((0, 1), (0, 2), (3, 4), (2, 3), (1, 6)):
    y[0, i, j] = 1
with T.patched((E, "group_concentration", T._mutant(E.group_concentration, "ok = rows & (mass > 0)", "ok = rows"))):
    ss, sa, sz = E.head_concentration(out, y)["short"]
ok = np.allclose(ss, [0.5 / 0.85, 1.0], atol=1e-6) if len(ss) == 2 else False
print("zero-mass guard removed:", "SURVIVED" if ok and sz == 1 else f"KILLED (short shares {ss}, zero {sz})")
