"""Stage D of the lane head (E273): sections = lane pieces = ids (one id per lane, new ids only at splits and merges), as a soft absorbing walk on the link matrix.

    section_membership(links, s_prob) -> M (B, K, K): M[b, n, k] = the probability that the section of active node n starts at active node k

Walk BACKWARDS from node n: with probability s(m) the node m is the first node of its section (the walk is absorbed, the section's id = m), otherwise it steps to a predecessor m' drawn from
Pb[m, .] = softmax over the incoming link logits of m and a "no predecessor" option (logit 0). With Q = diag(1 - s) Pb and R = diag(s):  M = (I - Q)^-1 R, rows normalised. A linear solve:
differentiable, no loop over the chain length. Nodes that are not active are absorbing (s = 1).
`section_loss(M, ...)`: the cross entropy of the true section start of every GT node + the same-section probability sum_k M[i, k] M[j, k] of sampled node pairs against the GT pieces.
Run the self-test: python sections.py
"""
import torch
import torch.nn.functional as F

from qnodes import N


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


def start_targets(gt, links):
    """for every active node: the active index of the first node of its piece (-1: not a GT node / the start is not among the active nodes) -> (B, K) long, and the piece id (B, K) (-1 none)"""
    idx = links["act"]
    B, K = idx.shape
    dev = idx.device
    piece = gt["piece"].flatten(1)                                                              # (B, 2500) piece index of each cell, -1 none
    sstart = gt["sstart"].flatten(1) > 0.5
    P = int(piece.max().item()) + 1 if piece.numel() else 1
    first = torch.full((B, max(P, 1)), -1, dtype=torch.long, device=dev)
    cells = torch.arange(N * N, device=dev)[None].expand(B, -1)
    sel = sstart & (piece >= 0)
    bb = torch.arange(B, device=dev)[:, None].expand_as(piece)
    first[bb[sel], piece[sel]] = cells[sel]
    inv = torch.full((B, N * N), -1, dtype=torch.long, device=dev)
    inv.scatter_(1, idx, torch.arange(K, device=dev)[None].expand(B, -1))
    pa = piece.gather(1, idx)                                                                   # (B, K)
    fc = first.gather(1, pa.clamp_min(0))                                                       # first cell of the piece of each node
    tgt = inv.gather(1, fc.clamp_min(0))
    tgt = torch.where((pa >= 0) & (fc >= 0), tgt, torch.full_like(tgt, -1))
    return tgt, pa


def section_loss(M, gt, links, n_pairs=2048):
    tgt, pa = start_targets(gt, links)
    valid = links["valid"] & (pa >= 0)
    ok = valid & (tgt >= 0)
    ce = torch.zeros((), device=M.device)
    if ok.any():
        pk = M.gather(2, tgt.clamp_min(0)[..., None])[..., 0]
        ce = -(torch.log(pk.clamp_min(1e-6)) * ok).sum() / ok.sum()
    # same-section probability of sampled pairs of GT nodes (half of the pairs from the same piece)
    B, K = valid.shape
    pl = torch.zeros((), device=M.device)
    cnt = 0
    for b in range(B):
        ids = valid[b].nonzero(as_tuple=True)[0]
        if len(ids) < 2:
            continue
        i = ids[torch.randint(len(ids), (n_pairs,), device=M.device)]
        j = ids[torch.randint(len(ids), (n_pairs,), device=M.device)]
        same = (pa[b, i] == pa[b, j]).float()
        p_same = (M[b, i] * M[b, j]).sum(-1).clamp(1e-6, 1 - 1e-6)
        w = 1.0 + 4.0 * same
        bce = -(same * torch.log(p_same) + (1 - same) * torch.log(1 - p_same))
        pl = pl + (bce * w).sum() / w.sum()
        cnt += 1
    return ce, pl / max(cnt, 1)


def _selftest():
    torch.manual_seed(0)
    B, K = 1, 12
    # two chains: nodes 0..5 (section start 0) and 6..11 (section start 6); true links i -> i+1 inside a chain
    valid = torch.ones(B, K, dtype=torch.bool)
    pb, pa, pc = [], [], []
    for a in list(range(0, 5)) + list(range(6, 11)):
        pb.append(0), pa.append(a), pc.append(a + 1)
    # false candidate links: 5 -> 6 (chains are not joined), 11 -> 0
    pb += [0, 0]
    pa += [5, 11]
    pc += [6, 0]
    logit = torch.tensor([4.0] * 10 + [-4.0, -4.0])
    links = dict(act=torch.arange(K)[None], valid=valid, pb=torch.tensor(pb), pa=torch.tensor(pa), pc=torch.tensor(pc), logit=logit.clone().requires_grad_(True))
    s_logit = torch.full((B, N, N), -4.0)
    s_logit.view(B, -1)[0, 0] = 4.0
    s_logit.view(B, -1)[0, 6] = 4.0
    M, Pb = section_membership(links, s_logit)
    start = M.argmax(-1)[0]
    assert start[:6].eq(0).all() and start[6:].eq(6).all(), start
    assert float(M[0, 5, 0]) > 0.9, M[0, 5, :8]
    # a gradient reaches the link logits and the section-start logits
    gt = dict(piece=torch.full((B, N, N), -1, dtype=torch.long), sstart=torch.zeros(B, N, N))
    gt["piece"].view(B, -1)[0, :6], gt["piece"].view(B, -1)[0, 6:12] = 0, 1
    gt["sstart"].view(B, -1)[0, 0], gt["sstart"].view(B, -1)[0, 6] = 1, 1
    sl = s_logit.clone().requires_grad_(True)
    M2, _ = section_membership(links, sl)
    ce, pl = section_loss(M2, gt, links)
    (ce + pl).backward()
    assert links["logit"].grad is not None and float(links["logit"].grad.abs().sum()) > 0 and float(sl.grad.abs().sum()) > 0
    print("start CE", float(ce), "pair loss", float(pl))
    assert float(ce) < 0.2
    print("sections selftest ok")


if __name__ == "__main__":
    _selftest()
