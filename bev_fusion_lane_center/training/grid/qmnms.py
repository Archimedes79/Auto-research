"""E292 (D82): the one-pass Matrix-NMS decay as a layer of the forward path on the 2 m node grid, the torch twin of `evaluation/qgraph_readout.matrix_nms` (E288: R 1.75 m, sigma 0.1 reproduces the hard close-node step of the delivered graph).

    e_damped_logit = matrix_nms_logit(e_logit (B, H, W), pos (B, 2, H, W) metres, radius=1.75, sigma=0.1)

For a cell i with the existence probability e_i, the stronger cells j are those of its 8 neighbours with e_j > e_i (a tie: the lower cell index is the stronger; a comparison without gradient, the order of the hard rule). Their overlap with i is
o_ij = relu(1 - d_ij / radius) of the predicted node positions; the compensation of j is c_j = max over the cells k stronger than j of o_jk; the decay of i is the minimum over its stronger neighbours of
exp(-(o_ij^2 - c_j^2) / sigma), capped at 1, and e'_i = e_i x decay_i. With radius <= 2 m only the 8 neighbour cells can overlap (a node lies within +-1 m of its cell centre, two cells apart the nodes are at least 2 m apart),
so this equals the numpy function on the nodes with e >= thr exactly (tested against it in `python qmnms.py`). Differentiable in the node's own e and in the positions (through the overlaps and the compensation). NOT in the probability of the stronger neighbour: the decay depends on the overlaps only, the probabilities decide the
order (a comparison without gradient), so unlike E277's product form the loss cannot raise the stronger node through the weaker one's decay. Returns the logit of e'.
"""
import torch
import torch.nn.functional as F


def matrix_nms_logit(e_logit, pos, radius=1.75, sigma=0.1, eps=1e-6):
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
    comp = torch.stack([torch.where(m, o, zero) for o, m in zip(ov, dom)], 0).max(0)[0]            # c_i: the largest overlap of cell i with a stronger neighbour
    cp = F.pad(comp[:, None], (1, 1, 1, 1), value=0.0)[:, 0]
    decay = torch.ones_like(e)
    for (dy, dx), o, m in zip(shifts, ov, dom):
        cj = cp[:, dy:dy + H, dx:dx + W]                                                            # the compensation of the stronger neighbour j
        f = torch.exp(-(o.square() - cj.square()) / sigma)
        decay = torch.minimum(decay, torch.where(m, f, torch.ones_like(f)))
    e2 = (e * decay).clamp(eps, 1.0 - eps)
    return torch.log(e2) - torch.log1p(-e2)


def _selftest():
    import os
    import sys
    import numpy as np
    root = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, os.path.join(root, "evaluation"))
    import qgraph_readout as QR
    torch.manual_seed(0)
    B, H, W = 3, 50, 50
    cell = torch.stack(torch.meshgrid(-50 + 2 * (torch.arange(H).float() + 0.5), -50 + 2 * (torch.arange(W).float() + 0.5), indexing="ij"), 0)[None]
    for radius, sigma in ((1.75, 0.1), (1.5, 0.5), (2.0, 0.2)):
        e_logit = torch.randn(B, H, W) * 2.5
        pos = cell + (torch.rand(B, 2, H, W) * 2 - 1)                                               # the node lies within +-1 m of its cell centre
        out = matrix_nms_logit(e_logit, pos, radius, sigma).sigmoid()
        for b in range(B):
            e = e_logit[b].sigmoid().flatten().numpy().astype(float)
            ref = QR.matrix_nms(pos[b].flatten(1).t().numpy().astype(float), e, radius, thr=0.5, sigma=sigma)       # 0 below thr after the decay
            mine = out[b].flatten().numpy().astype(float)
            sel = e >= 0.5
            ok = np.abs(np.where(ref > 0, ref, 0.0) - np.where(sel & (mine >= 0.5), mine, 0.0))[sel]
            assert ok.max() < 2e-5, (radius, sigma, ok.max())
    # a gradient reaches the weaker node's own probability and both positions (not the stronger node's probability: the order is a comparison)
    e_logit = torch.zeros(1, 4, 4)
    e_logit[0, 1, 1], e_logit[0, 1, 2] = 3.0, 2.0
    e_logit.requires_grad_(True)
    pos = (cell[:, :, :4, :4] + 0.0).clone().requires_grad_(True)
    pos.data[0, 1, 1, 2] -= 1.8                                                                     # pull the second node onto the first (4 m grid: cells 2 m apart, offset -1.8 -> 0.2 m)
    matrix_nms_logit(e_logit, pos).sigmoid()[0, 1, 2].backward()
    assert e_logit.grad[0, 1, 2].abs() > 0 and e_logit.grad[0, 1, 1].abs() == 0 and pos.grad[0, :, 1, 2].abs().sum() > 0 and pos.grad[0, :, 1, 1].abs().sum() > 0, (e_logit.grad[0, 1, 2], e_logit.grad[0, 1, 1], pos.grad[0, :, 1, 2], pos.grad[0, :, 1, 1])
    print("qmnms selftest ok")


if __name__ == "__main__":
    _selftest()
