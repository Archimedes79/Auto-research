"""E292 scratch: qmnms.matrix_nms_logit in log space (no 1e-6 clamp, no dead gradient). usage: python temp/_session_scratch_s292/patch_qmnms.py (writes the patched COPY new/qmnms.py)"""
import os
p = "training/grid/qmnms.py"
s = open(p, encoding="utf-8").read()
a = s[s.index("    decay = torch.ones_like(e)\n"):s.index("def _selftest():")]
b = '''    x = torch.zeros_like(e)                                                                         # decay_i = exp(-x_i), x_i = max(0, the largest (o_ij^2 - c_j^2) / sigma over the stronger neighbours j)
    for (dy, dx), o, m in zip(shifts, ov, dom):
        cj = cp[:, dy:dy + H, dx:dx + W]                                                            # the compensation of the stronger neighbour j
        x = torch.maximum(x, torch.where(m, (o.square() - cj.square()) / sigma, zero))
    # logit(e x decay) = log(decay) - log(1 - decay + exp(-z)), computed without leaving the log space: no clamp, so the gradient is alive at any logit (the first version clamped e' at 1e-6, which made
    # it flat beyond |logit| 13.8: the dead gradient that E277's critic removed from the soft step); where nothing decays (x = 0) the output is z itself
    z = e_logit.float()
    has = x > 0.0
    ld = torch.where(has, -x, torch.full_like(x, -1.0))                                             # a safe value where nothing decays: the formula is not evaluated there, so no inf * 0 in the backward pass
    damped = ld - torch.logaddexp(torch.log(-torch.expm1(ld)), -z)
    return torch.where(has, damped, z)


'''
s = s.replace(a, b)
assert "eps" in s
s = s.replace("def matrix_nms_logit(e_logit, pos, radius=1.75, sigma=0.1, eps=1e-6):", "def matrix_nms_logit(e_logit, pos, radius=1.75, sigma=0.1):")
open("temp/_session_scratch_s292/new/qmnms.py", "w", encoding="utf-8", newline="\n").write(s)
print("ok")

# the selftest: no dead gradient at the extremes, nothing NaN where nothing decays
s = open("temp/_session_scratch_s292/new/qmnms.py", encoding="utf-8").read()
a = '    print("qmnms selftest ok")\n'
assert s.count(a) == 1
b = ('    # no dead gradient at the extremes (the first version clamped e\' at 1e-6): an isolated node returns its own logit with gradient 1 at any |logit|; a node under a stronger one at the same place\n'
     '    # returns about z + log(decay) (decay = exp(-1 / sigma)) with a finite, positive gradient even for a logit of -14.5 (e below 1e-6)\n'
     '    for v in (-30.0, -14.5, 14.5, 30.0):\n'
     '        lz = torch.tensor([[[v, -9.0]]], requires_grad=True)\n'
     '        pz = torch.zeros(1, 2, 1, 2)\n'
     '        pz[0, 1, 0, 1] = 5.0                                                                        # the second cell is far away: nothing to decay it\n'
     '        oz = matrix_nms_logit(lz, pz)\n'
     '        oz.sum().backward()\n'
     '        assert torch.allclose(oz.detach()[0, 0], lz.detach()[0, 0], atol=1e-4) and torch.allclose(lz.grad[0, 0], torch.ones(2), atol=1e-4), (v, oz, lz.grad)\n'
     '    for v in (-30.0, -14.5, -5.0, 3.0):\n'
     '        lz = torch.tensor([[[8.0, v]]], requires_grad=True)                                         # the weaker node (cell 1) shares the stronger one\'s position\n'
     '        oz = matrix_nms_logit(lz, torch.zeros(1, 2, 1, 2), 1.75, 0.1)\n'
     '        oz[0, 0, 1].backward()\n'
     '        assert torch.isfinite(oz).all() and torch.isfinite(lz.grad).all() and float(lz.grad[0, 0, 1]) > 0.0, (v, oz, lz.grad)\n'
     '        assert v > -14.0 or abs(float(oz[0, 0, 1]) - (v - (1.0 - 1e-4 / 1.75) ** 2 / 0.1)) < 1e-3, (v, float(oz[0, 0, 1]))                  # e is tiny: logit(e x decay) = z + log(decay), decay = exp(-o^2 / sigma)\n'
     + a)
open("temp/_session_scratch_s292/new/qmnms.py", "w", encoding="utf-8", newline="\n").write(s.replace(a, b))
print("selftest extended")
