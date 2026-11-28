"""Stage D of the lane head (E273): sections = lane pieces = ids (one id per lane, new ids only at splits and merges), as a soft absorbing walk on the link matrix.

    section_membership(links, s_prob) -> M (B, K, K): M[b, n, k] = the probability that the section of active node n starts at active node k

Walk BACKWARDS from node n: with probability s(m) the node m is the first node of its section (the walk is absorbed, the section's id = m), otherwise it steps to a predecessor m' drawn from
Pb[m, .] = softmax over the incoming link logits of m and a "no predecessor" option (logit 0). With Q = diag(1 - s) Pb and R = diag(s):  M = (I - Q)^-1 R, rows normalised. A linear solve:
differentiable, no loop over the chain length. Nodes that are not active are absorbing (s = 1).
`section_loss(M, ...)`: the cross entropy of the true section start of every GT node + the same-section probability sum_k M[i, k] M[j, k] of sampled node pairs against the GT pieces.
Test: net/tests/test_sections.py
"""
import torch



def section_membership(links, s_logit, ridge=1e-3):
    """links: the dict of `qlinks.LinkHead` (act, valid, pb, pa, pc, logit); s_logit (B, N, N) the node head's section-start logits -> M (B, K, K), Pb (B, K, K)"""
    idx, valid = links["act"], links["valid"]
    B, K = valid.shape
    dev = idx.device
    s = s_logit.flatten(1).gather(1, idx).sigmoid().clamp(ridge, 1 - 1e-4)
    s = torch.where(valid, s, torch.ones_like(s))
    L = torch.full((B, K, K), float("-inf"), device=dev, dtype=links["logit"].dtype)
    L = L.index_put((links["pb"], links["pc"], links["pa"]), links["logit"])                   # row = the node, column = its candidate predecessor
    none = torch.zeros(B, K, 1, device=dev, dtype=L.dtype)
    Pb = torch.cat([L, none], -1).softmax(-1)[..., :K]
    Pb = Pb * valid[:, :, None] * valid[:, None, :]
    Q = (1 - s)[:, :, None] * Pb
    A = torch.eye(K, device=dev)[None] - Q
    M = torch.linalg.solve(A, torch.diag_embed(s))
    M = M.clamp_min(0)
    return M / M.sum(-1, keepdim=True).clamp_min(1e-9), Pb
