"""Test of the Matrix-NMS layer of the lane head (`net/model/qmnms.matrix_nms_logit`): it equals the hard numpy reference below (E288: the one-pass decay of the delivered graph) on the nodes with
e >= thr for three (radius, sigma) pairs; the gradient reaches the weaker node's own existence and both positions, not the stronger node's existence; no dead gradient at |logit| 30. CPU, seconds."""
import numpy as np
import torch

from qmnms import matrix_nms_logit


def matrix_nms(pos, e, radius, thr=0.5, sigma=0.5, protect=None):
    """E288: the one-pass Matrix-NMS decay (SOLOv2, Wang et al. 2020) on the nodes with e >= thr: the overlap of two nodes is o = max(0, 1 - d / radius); going through the nodes by e descending,
    the decay of node j is the minimum over the higher nodes i of exp(-(o_ij^2 - c_i^2) / sigma) with the compensation c_i = max over the nodes k above i of o_ki (a node that is itself suppressed
    suppresses less); the decayed score is e_j x decay_j. -> (M,) float: the decayed scores (0 below thr). No sequential keep set: one matrix product-free pass, differentiable in e and pos"""
    pos = np.asarray(pos, float).reshape(-1, 2)
    e = np.asarray(e, float).reshape(-1)
    out = np.zeros(len(e))
    cand = np.flatnonzero(e >= thr)
    if not len(cand):
        return out
    order = cand[np.argsort(-e[cand], kind="stable")]
    P = pos[order]
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    o = np.triu(np.maximum(0.0, 1.0 - d / radius), 1)                        # o[i, j], i above j
    comp = o.max(0)                                                         # c_i = max_k<i o[k, i] (column i holds the overlaps of i with the nodes above it)
    decay = np.exp(-(o ** 2 - comp[:, None] ** 2) / sigma)
    decay = np.where(np.triu(np.ones_like(o), 1) > 0, decay, np.inf).min(0)   # over the higher nodes i < j
    decay = np.where(np.isfinite(decay), decay, 1.0)
    if protect is not None:
        decay = np.where(np.asarray(protect, bool)[order], 1.0, decay)                                      # E296: a protected node is not decayed (it still decays the weaker ones)
    out[order] = e[order] * decay
    return np.where(out >= thr, out, 0.0)




def main():
    torch.manual_seed(0)
    B, H, W = 3, 50, 50
    cell = torch.stack(torch.meshgrid(-50 + 2 * (torch.arange(H).float() + 0.5), -50 + 2 * (torch.arange(W).float() + 0.5), indexing="ij"), 0)[None]
    for radius, sigma in ((1.75, 0.1), (1.5, 0.5), (2.0, 0.2)):
        e_logit = torch.randn(B, H, W) * 2.5
        pos = cell + (torch.rand(B, 2, H, W) * 2 - 1)                                               # the node lies within +-1 m of its cell centre
        out = matrix_nms_logit(e_logit, pos, radius, sigma).sigmoid()
        for b in range(B):
            e = e_logit[b].sigmoid().flatten().numpy().astype(float)
            ref = matrix_nms(pos[b].flatten(1).t().numpy().astype(float), e, radius, thr=0.5, sigma=sigma)       # 0 below thr after the decay
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
    # no dead gradient at the extremes (the first version clamped e' at 1e-6): an isolated node returns its own logit with gradient 1 at any |logit|; a node under a stronger one at the same place
    # returns about z + log(decay) (decay = exp(-1 / sigma)) with a finite, positive gradient even for a logit of -14.5 (e below 1e-6)
    for v in (-30.0, -14.5, 14.5, 30.0):
        lz = torch.tensor([[[v, -9.0]]], requires_grad=True)
        pz = torch.zeros(1, 2, 1, 2)
        pz[0, 1, 0, 1] = 5.0                                                                        # the second cell is far away: nothing to decay it
        oz = matrix_nms_logit(lz, pz)
        oz.sum().backward()
        assert torch.allclose(oz.detach()[0, 0], lz.detach()[0, 0], atol=1e-4) and torch.allclose(lz.grad[0, 0], torch.ones(2), atol=1e-4), (v, oz, lz.grad)
    for v in (-30.0, -14.5, -5.0, 3.0):
        lz = torch.tensor([[[8.0, v]]], requires_grad=True)                                         # the weaker node (cell 1) shares the stronger one's position
        oz = matrix_nms_logit(lz, torch.zeros(1, 2, 1, 2), 1.75, 0.1)
        oz[0, 0, 1].backward()
        assert torch.isfinite(oz).all() and torch.isfinite(lz.grad).all() and float(lz.grad[0, 0, 1]) > 0.0, (v, oz, lz.grad)
        assert v > -14.0 or abs(float(oz[0, 0, 1]) - (v - (1.0 - 1e-4 / 1.75) ** 2 / 0.1)) < 1e-3, (v, float(oz[0, 0, 1]))                  # e is tiny: logit(e x decay) = z + log(decay), decay = exp(-o^2 / sigma)
    print("test_qmnms ok: the layer equals the numpy reference on the nodes above the threshold; the gradient reaches the weaker node and both positions, none is dead at the extremes")


if __name__ == "__main__":
    main()
