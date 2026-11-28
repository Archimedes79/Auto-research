"""E292 (D82): the one-pass Matrix-NMS decay as a layer of the forward path on the 2 m node grid, the torch twin of the hard numpy step `matrix_nms` kept in
net/tests/test_qmnms.py (E288: R 1.75 m, sigma 0.1 reproduces the hard close-node step of the delivered graph).

    e_damped_logit = matrix_nms_logit(e_logit (B, H, W), pos (B, 2, H, W) metres, radius=1.75, sigma=0.1)

For a cell i with the existence probability e_i, the stronger cells j are those of its 8 neighbours with e_j > e_i (a tie: the lower cell index is the stronger;
a comparison without gradient, the order of the hard rule). Their overlap with i is
o_ij = relu(1 - d_ij / radius) of the predicted node positions; the compensation of j is c_j = max over the cells k stronger than j of o_jk; the decay of i is
the minimum over its stronger neighbours of
exp(-(o_ij^2 - c_j^2) / sigma), capped at 1, and e'_i = e_i x decay_i. With radius <= 2 m only the 8 neighbour cells can overlap (a node lies within +-1 m of
its cell centre, two cells apart the nodes are at least 2 m apart), so this equals the numpy function on the nodes with e >= thr exactly (tested against it in
net/tests/test_qmnms.py). Differentiable in the node's own e and in the positions (through the overlaps and the compensation). NOT in the probability of the
stronger neighbour: the decay depends on the overlaps only, the probabilities decide the order (a comparison without gradient), so unlike E277's product form
the loss cannot raise the stronger node through the weaker one's decay. Returns the logit of e'.
"""
import torch
import torch.nn.functional as F


def matrix_nms_logit(e_logit, pos, radius=1.75, sigma=0.1):
    assert 0.0 < radius <= 2.0 and sigma > 0.0, (radius, sigma)
    e = e_logit.float().sigmoid()
    pos = pos.float()
    B, H, W = e.shape
    idx = torch.arange(H * W, device=e.device).view(1, H, W)
    ep = F.pad(e[:, None], (1, 1, 1, 1), value=0.0)[:, 0]
    pp = F.pad(pos, (1, 1, 1, 1), value=0.0)
    ip = F.pad(idx[:, None].float(), (1, 1, 1, 1), value=float(H * W))[:, 0]
    shifts = [(dy, dx) for dy in (0, 1, 2) for dx in (0, 1, 2) if (dy, dx) != (1, 1)]
    ed = e.detach()
    ov, dom = [], []
    for dy, dx in shifts:
        ej = ep[:, dy:dy + H, dx:dx + W]
        pj = pp[:, :, dy:dy + H, dx:dx + W]
        ij = ip[:, dy:dy + H, dx:dx + W]
        d = ((pos - pj).square().sum(1) + 1e-8).sqrt()
        ov.append(F.relu(1.0 - d / radius))
        dom.append(((ej.detach() > ed) | ((ej.detach() == ed) & (ij < idx))) & (ij < H * W))
    zero = torch.zeros_like(e)
    # c_i: the largest overlap of cell i with a stronger neighbour
    comp = torch.stack([torch.where(m, o, zero) for o, m in zip(ov, dom)], 0).max(0)[0]
    cp = F.pad(comp[:, None], (1, 1, 1, 1), value=0.0)[:, 0]
    # decay_i = exp(-x_i), x_i = max(0, the largest (o_ij^2 - c_j^2) / sigma over the stronger neighbours j)
    x = torch.zeros_like(e)
    for (dy, dx), o, m in zip(shifts, ov, dom):
        cj = cp[:, dy:dy + H, dx:dx + W]                                                            # the compensation of the stronger neighbour j
        x = torch.maximum(x, torch.where(m, (o.square() - cj.square()) / sigma, zero))
    # logit(e x decay) = log(decay) - log(1 - decay + exp(-z)), computed without leaving the log space: no clamp, so the gradient is alive at any logit (the
    # first version clamped e' at 1e-6, which made it flat beyond |logit| 13.8: the dead gradient that E277's critic removed from the soft step); where nothing
    # decays (x = 0) the output is z itself
    z = e_logit.float()
    has = x > 0.0
    # a safe value where nothing decays: the formula is not evaluated there, so no inf * 0 in the backward pass
    ld = torch.where(has, -x, torch.full_like(x, -1.0))
    damped = ld - torch.logaddexp(torch.log(-torch.expm1(ld)), -z)
    return torch.where(has, damped, z)
