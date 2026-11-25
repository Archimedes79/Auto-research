"""Losses of the grid chain: the loss side of net/model/chain.py (D152; the functions are unchanged, `chain` holds the network)."""
import torch

import assign as AS
import losses as GL
import soft_lines as SL
import targets as TG


CUT = 1e-3                                                                      # weights below it are not sent through the assignment (numerical, see the module doc)


W_FUSED, W_ASSIGN = 1.0, 1.0


def assign_loss(model, fused, mps, gmasks, g05, t05, m05, d, idx, device):
    """the assignment loss of the cells of the fused map of the last frame (see the module doc); samples without a map contribute nothing -> (loss, road accuracy of the weight)"""
    B, _, n, _ = fused.shape
    pos = SL.positions(fused, 1.0, (-50.0, -50.0)).flatten(2)                    # (B, 2, n*n) metres, frame of the graphs
    hd = fused[:, 1:3].flatten(2)
    hd = hd / hd.norm(dim=1, keepdim=True).clamp_min(1e-6)
    ws = fused[:, 0].flatten(1)
    tot, wsum, hit, cnt = fused.new_zeros(()), fused.new_zeros(()), fused.new_zeros(()), fused.new_zeros(())
    for b in range(B):
        if mps[b] is None or gmasks[b] is None:
            continue
        sel = ws[b].detach() > CUT
        if int(sel.sum()) == 0:
            continue
        mp = AS.to_torch(mps[b], device)
        R = int(mp["n_ids"])
        out = model.assign_vertices(pos[b].T[sel], hd[b].T[sel], ws[b][sel], mp)
        P = out["P_road"]                                                       # (N, R + 1)
        gm = gmasks[b].to(device)
        near = idx[b].flatten()[sel]
        tgt = gm[near].clone()                                                  # (N, R + 1) plausible roads of the nearest GT point
        far = d[b].flatten()[sel] > 2.0
        tgt[far] = False
        tgt[far, R] = True                                                      # no lane within 2 m: the dustbin
        nll = -torch.log((P * tgt.to(P.dtype)).sum(1).clamp_min(1e-6))
        w = ws[b][sel].detach()
        tot = tot + (w * nll).sum() / (w.sum() + 1.0)
        wsum = wsum + 1.0
        hit = hit + (w * (P * tgt.to(P.dtype)).sum(1)).sum()
        cnt = cnt + w.sum()
    return tot / wsum.clamp_min(1.0), (hit / cnt.clamp_min(1e-6))


def chain_loss(model, batch, w_fused=W_FUSED, w_assign=W_ASSIGN, per100=GL.PER100):
    """-> (loss, info); batch of `chain_data.collate`; called inside the trainer's autocast, the chain parts after the head run in float32"""
    dev = next(model.parameters()).device
    x = batch["x"].to(dev, non_blocking=True)
    out = model(x, batch["step"].to(dev), batch["has"].to(dev))
    T = x.shape[1]
    g05, t05, m05 = batch["g05"].to(dev).float(), batch["t05"].to(dev).float(), batch["m05"].to(dev).float()
    info, loss = {}, 0.0
    if model.train_head:
        hl = 0.0
        for t in range(T):
            l_t, i_t = GL.grid_loss(out["outs"][t], dict(g05=g05[:, t], t05=t05[:, t], m05=m05[:, t]), per100=per100)
            hl = hl + l_t / T
            if t == T - 1:
                info.update({f"h_{k}": v for k, v in i_t.items() if k in ("cov", "prec", "vgt100")})
        loss = loss + hl
        info["head"] = float(hl)
    with torch.autocast(device_type="cuda", enabled=False):
        fused = out["fused"].float()
        with torch.no_grad():
            d, off_t, head_t, idx = TG.cell_targets(g05[:, -1], t05[:, -1], n=fused.shape[-1])
        st = GL.sparse_terms(fused, d, off_t, g05[:, -1], t05[:, -1], m05[:, -1], per100)
        l_fused = sum(GL.W[k] * st[k] for k in ("off", "cov", "prec", "bud"))
        loss = loss + w_fused * l_fused
        info.update({f"f_{k}": float(v) for k, v in st.items()})
        l_as, acc = assign_loss(model, fused, batch["mp"], batch["gmask"], g05[:, -1], t05[:, -1], m05[:, -1], d, idx, dev)
        loss = loss + w_assign * l_as
        info["assign"], info["road_mass"] = float(l_as), float(acc)
        info["vgt100"] = float((fused[:, 0].sum((1, 2)) / (m05[:, -1].sum(1) * 0.5).clamp_min(1.0)).mean() * 100)
    return loss, info
