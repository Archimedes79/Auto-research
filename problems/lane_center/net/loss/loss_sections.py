"""Losses of the section stage: the loss side of net/model/sections.py (D152; the functions are unchanged, `sections` holds the network)."""
import torch

from lane_center.net.model.qnodes import N


def start_targets(gt, links):
    """for every active node: the active index of the first node of its piece (-1: not a GT node / the start is not among the active nodes) -> (B, K) long, and
    the piece id (B, K) (-1 none)"""
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
