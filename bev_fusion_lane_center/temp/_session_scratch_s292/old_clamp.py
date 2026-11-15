import os, sys, torch
sys.path.insert(0, os.environ["BFLC_ROOT"] + "/temp/_session_scratch_s292")
import qmnms_old as OLD
import qmnms as NEW
for v in (-20.0, -14.5, -13.0, 13.0, 14.5, 20.0):
    out = []
    for f in (OLD.matrix_nms_logit, NEW.matrix_nms_logit):
        z = torch.tensor([[[v, -9.0]]], requires_grad=True)
        p = torch.zeros(1, 2, 1, 2); p[0, 1, 0, 1] = 5.0
        o = f(z, p); o[0, 0, 0].backward()
        out.append((round(float(o[0, 0, 0]), 3), round(float(z.grad[0, 0, 0]), 3)))
    print(f"isolated node, logit {v:6.1f}: old (out, grad) {out[0]}   new {out[1]}")
