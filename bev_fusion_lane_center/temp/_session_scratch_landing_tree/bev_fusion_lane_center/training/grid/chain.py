"""The grid chain as ONE differentiable function (E272): frames -> head -> sparsify -> hold over time -> soft assignment to the SD map, and the one loss it is trained with.

    GridChain(head, hold, assigner).forward(x (B, T, 39, 200, 200), step (B, T - 1, 3), has (B, T - 1)) -> dict(outs (T of (B, 6, 100, 100)), sparse (T of (B, 5, 100, 100)), fused (B, 5, 100, 100))
    chain_loss(model, batch) -> (loss, info)

Per frame: the head (stage 1) and `soft_lines.sparsify` (stage 2). Over the frames of a chain: `hold.Hold` warps the fused map of the previous frame with the ego motion and fuses it with
the current sparse map (stage 3; a missing predecessor, has = 0, passes the current map through). The assignment (stage 4) reads the cells of the FUSED map of the last frame as vertices
(position = cell centre + offset, heading, weight) and gives each a distribution over the SD-map roads and a dustbin. No argmax, no matching, no top-k, no threshold on this path
(the assignment is computed for the cells with a weight above 1e-3: a numerical cut, those cells carry less than 0.1 % of the loss weight of a frame).
Loss = mean over the frames of the head loss (`losses.grid_loss`) + the sparse-map terms on the FUSED map (`losses.sparse_terms`: offset, coverage, precision, vertex budget) + the assignment
loss: -log of the probability mass on the plausible roads of the nearest GT point (the dustbin for cells farther than 2 m from any lane or without a road), weighted by the cell weight
(detached: the weights are trained by the sparse terms, the assignment loss trains the assigner and moves the vertices' positions and headings).
`model.train_head = False` runs the head under no_grad and drops the per-frame head loss: the stage in which only the hold and the assigner are fitted.
Run the self-test: python chain.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training", "grid"))
import assign as AS  # noqa: E402
import head as GH  # noqa: E402
import hold as HO  # noqa: E402
import losses as GL  # noqa: E402
import soft_lines as SL  # noqa: E402
import targets as TG  # noqa: E402

CUT = 1e-3                                                                      # weights below it are not sent through the assignment (numerical, see the module doc)
W_FUSED, W_ASSIGN = 1.0, 1.0


class GridChain(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.head = GH.GridHead(width=width)
        self.hold = HO.Hold()
        self.assigner = AS.Assigner()
        self.train_head = True

    def forward(self, x, step, has=None):
        B, T = x.shape[:2]
        outs, sps = [], []
        for t in range(T):
            if self.train_head:
                o = self.head(x[:, t].float())
            else:
                with torch.no_grad():
                    o = self.head(x[:, t].float())
            with torch.autocast(device_type="cuda", enabled=False):
                sps.append(SL.sparsify(o.float()))
            outs.append(o)
        fused, age = sps[0], torch.zeros_like(sps[0][:, :1])
        with torch.autocast(device_type="cuda", enabled=False):
            for t in range(1, T):
                new, age_n = self.hold(sps[t], fused, age, step[:, t - 1].float())
                if has is not None:
                    h = has[:, t - 1].view(-1, 1, 1, 1)
                    new, age_n = h * new + (1 - h) * sps[t], h * age_n
                fused, age = new, age_n
        return dict(outs=outs, sparse=sps, fused=fused)

    def assign_vertices(self, V, H, w, mp):
        """vertices V (N, 2) metres in the frame of the graphs (x right, y forward), headings H (N, 2), weights w (N,), map tokens mp (torch, `assign.to_torch`) -> the Assigner's dict"""
        pos = torch.stack([V[:, 1], -V[:, 0]], 1)                              # -> grid frame (x forward, y left)
        hdg = torch.stack([H[:, 1], -H[:, 0]], 1)
        return self.assigner(pos, hdg, w, mp)


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


def _selftest():
    torch.manual_seed(0)
    np.random.seed(0)
    import chain_data as CD
    B, T, n = 2, 3, 100
    ys = torch.linspace(-40, 39.5, 160)
    g = torch.stack([torch.full((160,), 0.2), ys], 1)                           # a lane along y (forward) at x = 0.2, metric frame
    t = torch.tensor([0.0, 1.0]).expand(160, 2)
    sd = dict(roads=[(np.array([[0.2, -50.0], [0.2, 50.0]]), np.array([1.0, 1.0])), (np.array([[-50.0, 30.0], [50.0, 30.0]]), np.array([1.0, 1.0]))], route=[])
    mp = AS.map_tokens(sd)
    gm = torch.from_numpy(CD.road_labels(g.numpy().astype(np.float64), t.numpy().astype(np.float64), mp))
    assert gm[:, 0].all() and not gm[:, 1].any(), "the lane belongs to road 0 only"
    batch = dict(x=torch.randn(B, T, 39, 200, 200), step=torch.tensor([[[0.0, 4.0, 0.0]] * (T - 1)] * B), has=torch.ones(B, T - 1),
                 g05=g[None, None].repeat(B, T, 1, 1), t05=t[None, None].repeat(B, T, 1, 1), m05=torch.ones(B, T, 160), mp=[mp, None], gmask=[gm, None])
    model = GridChain(width=16)
    loss, info = chain_loss(model, batch)
    loss.backward()
    grads = {k: sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None) for k, m in (("head", model.head), ("hold", model.hold), ("assign", model.assigner))}
    print("G2 gradient per block", {k: f"{v:.2e}" for k, v in grads.items()}, "| terms", {k: round(v, 3) for k, v in info.items()})
    assert all(v > 0 for v in grads.values()), "the loss does not reach every block"
    # the head reaches through the assignment: a loss of the assignment alone moves the head
    model.zero_grad()
    out = model(batch["x"], batch["step"], batch["has"])
    with torch.no_grad():
        d, off_t, head_t, idx = TG.cell_targets(batch["g05"][:, -1], batch["t05"][:, -1])
    la, _ = assign_loss(model, out["fused"], batch["mp"], batch["gmask"], None, None, None, d, idx, "cpu")
    la.backward()
    assert sum(float(p.grad.abs().sum()) for p in model.head.parameters() if p.grad is not None) > 0, "the assignment loss does not reach the head"
    # frozen head: no gradient into it, the hold and the assigner still learn
    model.zero_grad()
    model.train_head = False
    loss2, _ = chain_loss(model, batch)
    loss2.backward()
    assert all(p.grad is None for p in model.head.parameters()), "a frozen head got a gradient"
    assert sum(float(p.grad.abs().sum()) for p in model.hold.parameters() if p.grad is not None) > 0
    # has = 0 passes the current map through
    model.train_head = True
    o = model(batch["x"], batch["step"], torch.zeros(B, T - 1))
    assert torch.allclose(o["fused"], o["sparse"][-1], atol=1e-6), "without predecessors the fused map is the current one"
    print("chain selftest ok")


if __name__ == "__main__":
    _selftest()
