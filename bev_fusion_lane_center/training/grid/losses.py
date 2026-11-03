"""Grid stages 1 + 2: the losses on the head output and on the sparse map; matching-free, every term differentiable.

batch: g05 (B, G, 2) GT lane points every 0.5 m in metres (padded with 1e4), t05 (B, G, 2) their travel tangents; m05 (B, G) the real-point mask.
out:   head output (B, 6, 100, 100) (`head.GridHead`); sparse = `soft_lines.sparsify(out)`.

L_p     BCE against the soft centre-lane target exp(-d^2 / 2 sigma^2) (sigma 1 m, d = cell centre to the nearest lane point) + soft Dice
L_dir   (1 - cos) between the heading and the tangent of the nearest lane point, weighted by the target
L_off   L1 of the sub-cell offset against nearest point - cell centre (cells, clipped to +-0.5), weighted by the target
L_cov   coverage of the GT points by the vertex weights, noisy-or: C(g) = 1 - prod_c (1 - w_c K(g, c)), K = exp(-across^2 / 2 (1 m)^2 - along^2 / 2 (4 m)^2) in the
        frame of the GT tangent, cell position = centre + offset; loss = mean(1 - C): a vertex pays for itself where it covers lane that nobody else covers
L_prec  mean over the vertex weight of the clipped distance (2 m) of the vertex to the nearest lane point: weight off the lane is punished
L_bud   the vertex budget (`soft_lines.budget`), only the excess over `per100` vertices per 100 m of GT lane
Run the self-test: python losses.py
"""
import torch
import torch.nn.functional as F

import soft_lines as SL
import targets as TG

W = dict(p=1.0, dice=1.0, dir=0.5, off=1.0, cov=2.0, prec=1.0, bud=1.0)
SIGMA = 1.0
PER100 = 12.0           # the budget the loss asks for (vertices per 100 m of GT lane); the read-out threshold is fixed separately on TRAIN


def grid_loss(out, batch, w=W, per100=PER100, sigma=SIGMA, k_along=4.0, k_across=1.0):
    """-> (loss, dict of floats)"""
    with torch.autocast(device_type="cuda", enabled=False):
        out = out.float()
        g05, t05, m05 = batch["g05"].float(), batch["t05"].float(), batch["m05"].float()
        B, _, n, _ = out.shape
        with torch.no_grad():
            d, off_t, head_t, _ = TG.cell_targets(g05, t05, n=n)
            T = TG.heat(d, sigma)
        logit = out[:, 0]
        l_p = F.binary_cross_entropy_with_logits(logit, T)
        pr = torch.sigmoid(logit)
        dice = 1 - (2 * (pr * T).sum((1, 2)) + 1) / (pr.sum((1, 2)) + T.sum((1, 2)) + 1)
        wt = T / (T.sum((1, 2), keepdim=True) + 1e-6)
        h = out[:, 1:3] / torch.sqrt((out[:, 1:3] ** 2).sum(1, keepdim=True) + 1e-6)
        l_dir = ((1 - (h * head_t).sum(1)) * wt).sum((1, 2))
        sp = SL.sparsify(out)
        l_off = ((sp[:, 3:5] - off_t.clamp(-0.5, 0.5)).abs().sum(1) * wt).sum((1, 2))
        wv = sp[:, 0]
        pos = SL.positions(sp, 1.0, (-50.0, -50.0)).flatten(2)                       # (B, 2, n*n) metres
        pts = g05[:, ::2]                                                           # one lane point per metre
        tan = t05[:, ::2]
        ok = m05[:, ::2]
        l_cov = []
        for b in range(B):
            rel = pos[b].T[None, :, :] - pts[b][:, None, :]                          # (G, n*n, 2)
            lon = (rel * tan[b][:, None, :]).sum(-1)
            lat = rel[..., 0] * tan[b][:, None, 1] - rel[..., 1] * tan[b][:, None, 0]
            K = torch.exp(-(lat / k_across) ** 2 / 2 - (lon / k_along) ** 2 / 2)
            logsurv = torch.log1p(-(wv[b].flatten()[None, :] * K).clamp(max=0.999)).sum(1)       # log prod (1 - w K)
            miss = torch.exp(logsurv)                                                  # 1 - C
            l_cov.append((miss * ok[b]).sum() / ok[b].sum().clamp_min(1.0))
        l_cov = torch.stack(l_cov)
        dc = d.clamp(max=2.0)
        l_prec = (wv * dc).sum((1, 2)) / (wv.sum((1, 2)) + 1.0)
        lines_m = m05.sum(1) * 0.5
        l_bud = SL.budget(sp, lines_m, per100)
        terms = dict(p=l_p, dice=dice.mean(), dir=l_dir.mean(), off=l_off.mean(), cov=l_cov.mean(), prec=l_prec.mean(), bud=l_bud.mean())
        loss = sum(w[k] * v for k, v in terms.items())
        info = {k: float(v) for k, v in terms.items()}
        info["vgt100"] = float((wv.sum((1, 2)) / lines_m.clamp_min(1.0)).mean() * 100)     # the soft vertex mass per 100 m (before the read-out)
    return loss, info


def _selftest():
    torch.manual_seed(0)
    B, n = 2, 100
    ys = torch.linspace(-40, 39.5, 160)
    g = torch.stack([torch.full((160,), 0.2), ys], 1)                                    # a lane along y at x = 0.2 (not between two cells: that is a tie of the ridge)
    t = torch.tensor([0.0, 1.0]).expand(160, 2)
    batch = dict(g05=g[None].repeat(B, 1, 1), t05=t[None].repeat(B, 1, 1), m05=torch.ones(B, 160))
    # the perfect output must have a lower loss than the zero output and the shifted one
    from_oracle = torch.zeros(B, 6, n, n)
    d, off, head, _ = TG.cell_targets(batch["g05"], batch["t05"])
    p = TG.heat(d).clamp(1e-4, 1 - 1e-4)
    from_oracle[:, 0] = torch.log(p / (1 - p))
    from_oracle[:, 1:3] = head
    from_oracle[:, 3:5] = torch.atanh(off.clamp(-0.5, 0.5) * 2 * 0.999)
    gate = torch.full((B, n, n), -6.0)
    gate[:, 50, 5::9] = 6.0                                                           # a vertex every 9 m on the lane (cells x = 50 -> -50 + 50.5 = 0.5 m)
    from_oracle[:, 5] = gate
    l_or, i_or = grid_loss(from_oracle, batch)
    zero = torch.zeros(B, 6, n, n)
    l_zero, i_zero = grid_loss(zero, batch)
    shifted = torch.roll(from_oracle, 5, dims=2)
    l_sh, i_sh = grid_loss(shifted, batch)
    print("loss oracle", float(l_or), "zero", float(l_zero), "shifted 5 m", float(l_sh), "| oracle terms", {k: round(v, 3) for k, v in i_or.items()})
    sparse_gate = from_oracle.clone()
    sparse_gate[:, 5] = -6.0
    sparse_gate[:, 5, 50, 5::30] = 6.0                                                 # one vertex every 30 m: the coverage must be worse than every 9 m
    _, i_sg = grid_loss(sparse_gate, batch)
    print("cov miss: every 9 m", round(i_or["cov"], 3), "every 30 m", round(i_sg["cov"], 3))
    assert l_or < l_zero and l_or < l_sh and i_or["cov"] < i_sg["cov"] and i_or["cov"] < 0.35 and i_or["p"] < 0.1
    # every output channel gets a gradient
    o = (from_oracle.clone() + 0.1 * torch.randn_like(from_oracle)).requires_grad_(True)
    loss, _ = grid_loss(o, batch)
    loss.backward()
    g_ch = o.grad.abs().sum((0, 2, 3))
    print("gradient per channel", [f"{v:.2e}" for v in g_ch.tolist()])
    assert (g_ch > 0).all()
    print("losses selftest ok")


if __name__ == "__main__":
    _selftest()
