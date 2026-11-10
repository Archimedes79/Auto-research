"""Loss extras from the literature scan of 2026-10-26 (plan.md 8.4; notes in literature/papers, scout tables
temp/scout_tables/papers_new_{mtl,mapl,trainrecipe}.csv): the pieces of the training loop that are not part of BEVLoss
(UWSO, group_gradient_cosines and cross_time_consistency are used by training/train.py; losses.py imports the weighted-sum
helpers `_wsum` / `_wcount` from here). Conventions as losses.py: prob / target (B, K, X, Y), weights w broadcastable (a cell
of weight 0 is ignored), per-channel results (K,), 0 for a channel without GT.

1  UWSO                    UW-SO (2408.07985): weights softmax(1 / (L.detach() T)) over the task groups, one
                           hyper-parameter T, no extra pass; drop-in for losses.TaskWeights (`loss.task = UWSO(...)`).
2  group_gradient_cosines  diagnostic: cosine and norm of the task-group gradients at the shared BEV feature (head
                           backward passes only) - look at it before any gradient surgery.
3  cross_time_consistency  train-only cross-time consistency of the predictions at dt = 0 and dt = +-0.5 s (MapTCL
                           2608.05209, ST-P3 / FIERY-style warping, UnFlow 1711.07837 masks): the prediction at dt = 0 is
                           carried to the other time by the splat geometry (ego steps + the motion field) and compared,
                           BOTH directions (forward: splat P_0 and compare with P_b, stop-grad on P_b; backward: gather
                           P_b at the carried positions and compare with P_0, stop-grad on P_0), only where both sides are
                           visible, cells without a counterpart cost a constant (no trivial solution by pushing everything
                           out of the grid), age weights. Nothing at inference.
Run the self-test: python training/loss_extras.py
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .lift_pd import Grid
    from .splat import chain_positions, grid_points, splat
except ImportError:                                   # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lift_pd import Grid
    from splat import chain_positions, grid_points, splat

EPS = 1e-6


def _wsum(x, w):
    return (x * w).sum((0, 2, 3))


def _wcount(w, like):
    return w.expand_as(like).sum((0, 2, 3)).clamp(min=EPS)


# ------------------------------------------------------------------------------------------------ 1
class UWSO(nn.Module):
    """Task-group weights w = softmax(1 / (L.detach() T)) (UW-SO, 2408.07985); total = scale * sum_g w_g L_g with
    scale = the number of groups present (so equal losses give the weight 1 per group, like equal weighting).
    Same call and return as losses.TaskWeights: forward(group_losses) -> (total, {group: weight})."""

    def __init__(self, T=1.0, scale=True):
        super().__init__()
        self.T, self.scale = T, scale

    def forward(self, group_losses):
        names = list(group_losses)
        L = torch.stack([group_losses[g] for g in names])
        w = torch.softmax(1.0 / (L.detach().clamp(min=EPS) * self.T), 0)
        total = (w * L).sum() * (len(names) if self.scale else 1.0)
        return total, {g: w[i].detach() for i, g in enumerate(names)}


# ------------------------------------------------------------------------------------------------ 2
def group_gradient_cosines(group_losses, feature):
    """Gradients of every group loss at the shared feature tensor (retain_graph): -> (names, cosine matrix (G, G),
    norms (G,)). Costs the head backward passes only. Negative cosines = conflicting groups."""
    names = list(group_losses)
    grads = []
    for g in names:
        gr = torch.autograd.grad(group_losses[g], feature, retain_graph=True, allow_unused=True)[0]
        grads.append(torch.zeros_like(feature).flatten() if gr is None else gr.flatten())
    G = torch.stack(grads)
    norms = G.norm(dim=1)
    unit = G / norms.clamp(min=EPS).unsqueeze(1)
    return names, unit @ unit.T, norms


# ------------------------------------------------------------------------------------------------ 3
def _norm_xy(pos, grid):
    """positions in metres (B, N, 2) -> grid_sample coordinates (B, N, 2) = (along W = y, along H = x)."""
    nx = 2 * (pos[..., 0] - grid.x_min) / (grid.nx * grid.res) - 1
    ny = 2 * (pos[..., 1] - grid.y_min) / (grid.ny * grid.res) - 1
    return torch.stack([ny, nx], -1)


def mass_at(pos, mass, grid):
    """The splat mass map (1, 1, X, Y) gathered at positions pos (1, N, 2) -> (1, 1, X, Y) (the mass at each cell's destination)."""
    X, Y = mass.shape[-2:]
    g = _norm_xy(pos, grid).view(1, X, Y, 2)
    return F.grid_sample(mass, g, mode="bilinear", padding_mode="zeros", align_corners=False)


def cross_time_consistency(probs, steps, v_hat, grid, weights=None, channels=None, mass_thr=0.5, penalty=0.5,
                           age_tau=1.0, importance_scale=6.0, v_tol=2.0):
    """probs {dt: (B, K, X, Y) probabilities}, dt = 0.0 required; steps {dt: [per sample: list of (dt_i, v_i, omega_i)
    from the time 0 to dt]} (python floats, as the GT meta['steps']); v_hat (B, 2, X, Y) motion field at time 0 (m/s,
    the axes of frame 0); weights {dt: (B, 1, X, Y)} visibility of the cells at that time (default 1; 0 = ignore);
    channels: which channels of probs to compare (default all). Returns {'total', 'fwd@dt', 'bwd@dt', 'matched@dt'}.
    forward: P_0 splatted to dt (stop-grad on P_dt); backward: P_dt gathered at the positions the cells of time 0 move to
    (stop-grad on P_0); a cell without counterpart (left the grid, mass < mass_thr, not visible) costs `penalty`
    instead of an error (the trivial solution 'push everything out of the grid' is not rewarded); pair weight
    exp(-|dt| / age_tau); in the forward splat the occupied cells win a collision (importance_scale * max probability,
    as in the model's window splat), otherwise a moved object is averaged with the static source of its destination;
    in the backward direction a cell whose destination is won by a source with a different velocity (more than v_tol
    m/s) is occluded and ignored (forward-backward check)."""
    P0 = probs[0.0]
    sel = list(range(P0.shape[1])) if channels is None else list(channels)
    B, _, X, Y = P0.shape
    pts = grid_points(grid, P0.device, P0.dtype).unsqueeze(0)
    one = torch.ones(B, 1, X, Y, device=P0.device, dtype=P0.dtype)
    out, total, wsum = {}, 0.0, 0.0
    for dt in sorted(k for k in probs if k != 0.0):
        Pb = probs[dt]
        w0, wb = (weights or {}).get(0.0, one), (weights or {}).get(dt, one)
        fwd, bwd, matched, ferr, berr = [], [], [], [], []
        for b in range(B):
            vo = v_hat[b:b + 1].permute(0, 2, 3, 1).reshape(1, X * Y, 2)
            pos = chain_positions(pts, steps[dt][b], vo)                                   # (1, N, 2) cells of time 0 at time dt
            # forward: splat P_0 (and the visibility of its cells) to the time dt
            K = len(sel)
            vcell = v_hat[b:b + 1].detach().reshape(1, 2, -1)
            feat = torch.cat([P0[b:b + 1, sel].reshape(1, K, -1), w0[b:b + 1].reshape(1, 1, -1), vcell], 1)
            imp = importance_scale * P0[b:b + 1, sel].amax(1).detach().reshape(1, -1)        # occupied cells win a collision
            sp, mass = splat(feat, pos, grid, importance=imp)
            warped, w_src, v_dst = sp[:, :K], sp[:, K:K + 1], sp[:, K + 1:K + 3]              # v_dst: velocity of the winner
            known = mass >= mass_thr
            # a counterpart that exists but is not supervised at time 0 (w_src <= 0.5) is ignored, not an error; a cell
            # with no source at all (mass < mass_thr) costs the constant
            valid = (wb[b:b + 1] > 0.5) & ~(known & (w_src <= 0.5))
            ok = known & valid                                                              # known + visible on both sides
            tgt = Pb[b:b + 1, sel].detach()
            err = (warped - tgt).abs().mean(1, keepdim=True)
            fwd.append(((err * ok).sum() + penalty * (valid & ~ok).sum()) / valid.sum().clamp(min=1))
            matched.append((ok.sum() / valid.sum().clamp(min=1)).detach())
            ferr.append(((err * ok).sum() / ok.sum().clamp(min=1)).detach())
            # backward: gather P_dt at the carried positions and compare with P_0
            g = _norm_xy(pos, grid).view(1, X, Y, 2)
            src = torch.cat([Pb[b:b + 1, sel], wb[b:b + 1], v_dst.detach()], 1)
            got = F.grid_sample(src, g, mode="bilinear", padding_mode="zeros", align_corners=False)
            inside = ((pos[..., 0] >= grid.x_min) & (pos[..., 0] <= grid.x_max) & (pos[..., 1] >= grid.y_min)
                      & (pos[..., 1] <= grid.y_max)).view(1, 1, X, Y)
            # forward-backward check (UnFlow-style occlusion): a cell of time 0 is visible at the time dt only if the
            # winner at its destination moves like it does; a static background cell covered by a moving object is
            # occluded - ignored, not an error
            v_here = v_hat[b:b + 1].detach()
            occluded = inside & (mass_at(pos, mass, grid) >= mass_thr) & ((got[:, K + 1:K + 3] - v_here).norm(dim=1, keepdim=True) > v_tol)
            ignored = inside & (got[:, K:K + 1] <= 0.5)                 # the counterpart exists but is not supervised
            valid0 = (w0[b:b + 1] > 0.5) & ~occluded & ~ignored         # left the grid = valid (costs the constant)
            ok_b = inside & valid0
            err_b = (got[:, :K] - P0[b:b + 1, sel].detach()).abs().mean(1, keepdim=True)    # stop-grad on P_0
            bwd.append(((err_b * ok_b).sum() + penalty * (valid0 & ~ok_b).sum()) / valid0.sum().clamp(min=1))
            berr.append(((err_b * ok_b).sum() / ok_b.sum().clamp(min=1)).detach())
        f, bk = torch.stack(fwd).mean(), torch.stack(bwd).mean()
        a = math.exp(-abs(dt) / age_tau)
        out[f"fwd@{dt:+.1f}"], out[f"bwd@{dt:+.1f}"], out[f"matched@{dt:+.1f}"] = f, bk, torch.stack(matched).mean()
        out[f"ferr@{dt:+.1f}"], out[f"berr@{dt:+.1f}"] = torch.stack(ferr).mean(), torch.stack(berr).mean()   # matched cells only
        total, wsum = total + a * 0.5 * (f + bk), wsum + a
    out["total"] = total / max(wsum, EPS)
    return out


# ----------------------------------------------------------------------------------------------- self-test
def _blob(grid, cx, cy, s=1.2, device="cpu"):
    xs, ys = grid.xy_centres(device)
    return torch.exp(-(((xs.view(-1, 1) - cx) ** 2 + (ys.view(1, -1) - cy) ** 2) / (2 * s * s)))[None, None]


def _selftest():
    torch.manual_seed(0)
    # 1 UW-SO: weights sum to 1, the smaller loss gets the larger weight, drop-in call of TaskWeights, equal losses -> 1 each
    uw = UWSO(T=1.0)
    tot, wts = uw({"areas": torch.tensor(0.2), "lines": torch.tensor(0.8), "objects": torch.tensor(0.5)})
    assert abs(sum(float(v) for v in wts.values()) - 1) < 1e-6 and wts["areas"] > wts["objects"] > wts["lines"]
    tot2, wts2 = uw({"a": torch.tensor(0.5), "b": torch.tensor(0.5)})
    assert abs(float(tot2) - 1.0) < 1e-6 and abs(float(wts2["a"]) - 0.5) < 1e-6, "equal losses: the sum"
    la = torch.tensor(0.4, requires_grad=True)
    uw({"a": la, "b": torch.tensor(1.0)})[0].backward()
    assert la.grad is not None and torch.isfinite(la.grad)

    # 2 group gradient cosines: parallel, opposite and orthogonal groups
    f = torch.randn(4, 3, requires_grad=True)
    g_par, g_opp, g_orth = (f * 2).sum(), (-f * 3).sum(), (f[:, 0] * 0 + f[:, 1].sum() * 0 + (f * torch.tensor([[0.0, 0.0, 1.0]]).expand_as(f)).sum() * 0).sum()
    base = (f * torch.tensor([[1.0, 0.0, 0.0]])).sum()                          # gradient (1, 0, 0) per row
    orth = (f * torch.tensor([[0.0, 1.0, 0.0]])).sum()
    names, C, nrm = group_gradient_cosines({"base": base, "par": base * 2, "opp": -base, "orth": orth}, f)
    i = {n: k for k, n in enumerate(names)}
    assert abs(C[i["base"], i["par"]] - 1) < 1e-6 and abs(C[i["base"], i["opp"]] + 1) < 1e-6 and abs(C[i["base"], i["orth"]]) < 1e-6
    assert abs(nrm[i["par"]] / nrm[i["base"]] - 2) < 1e-5

    # 3 cross-time consistency: a blob moving at 8 m/s, ego static, the probabilities at dt = 0 and +-0.5 s exact
    grid = Grid(x_min=0.0, x_max=32.0, y_min=-8.0, y_max=8.0, res=0.5)
    X, Y = grid.nx, grid.ny
    P = {0.0: _blob(grid, 12.0, -2.0), 0.5: _blob(grid, 16.0, -2.0), -0.5: _blob(grid, 8.0, -2.0)}
    st = {0.5: [[(0.5, 0.0, 0.0)]], -0.5: [[(-0.5, 0.0, 0.0)]]}
    v_true = torch.zeros(1, 2, X, Y)
    v_true[:, 0] = 8.0 * (P[0.0][:, 0] > 0.02).float()                          # the field is nonzero on the object only
    r = cross_time_consistency(P, st, v_true, grid)
    r0 = cross_time_consistency(P, st, torch.zeros(1, 2, X, Y), grid)           # no motion: the blob does not match
    assert r["total"] < 0.5 * r0["total"] and r["matched@+0.5"] > 0.9, (float(r["total"]), float(r0["total"]))
    # trivial solution: a motion field that pushes every cell out of the grid costs the constant penalty, not 0
    far = torch.zeros(1, 2, X, Y)
    far[:, 0] = 400.0
    rf = cross_time_consistency(P, st, far, grid, penalty=0.5)
    assert rf["total"] > 0.4 and rf["matched@+0.5"] < 0.05, float(rf["total"])
    # ego motion exact: a static pattern seen from a moved ego (v = 10 m/s, omega = 0.3 rad/s, dt = 0.5 s): P_dt rendered
    #   from the exact motion; with v_hat = 0 the consistency is small, with a wrong ego step it is not
    xs, ys = grid.xy_centres()
    v_e, w_e, dte = 10.0, 0.3, 0.5
    th = w_e * dte
    dxe, dye = v_e / w_e * math.sin(th), v_e / w_e * (1 - math.cos(th))

    def world(x, y):                                                            # a smooth static world, ego frame at t = 0
        return torch.exp(-(((x - 16.0) ** 2 + (y - 1.0) ** 2) / 8.0)) + 0.6 * torch.exp(-(((x - 8.0) ** 2 + (y + 3.0) ** 2) / 6.0))
    gx, gy = xs.view(-1, 1).expand(X, Y), ys.view(1, -1).expand(X, Y)
    qx, qy = gx - dxe, gy - dye
    # the cell (gx, gy) of the frame at dt sees the world point R(th) (gx, gy) + d in the frame at 0
    wx, wy = math.cos(th) * gx - math.sin(th) * gy + dxe, math.sin(th) * gx + math.cos(th) * gy + dye
    Pe = {0.0: world(gx, gy)[None, None], 0.5: world(wx, wy)[None, None]}
    rst = cross_time_consistency(Pe, {0.5: [[(0.5, v_e, w_e)]]}, torch.zeros(1, 2, X, Y), grid)
    rbad = cross_time_consistency(Pe, {0.5: [[(0.5, 0.0, 0.0)]]}, torch.zeros(1, 2, X, Y), grid)
    # (the totals contain the constant penalty for the cells that enter the field of view through the ego motion: judge the
    #  error over the matched cells)
    assert rst["ferr@+0.5"] < 0.5 * rbad["ferr@+0.5"] and rst["berr@+0.5"] < 0.5 * rbad["berr@+0.5"], \
        (float(rst["ferr@+0.5"]), float(rbad["ferr@+0.5"]), float(rst["berr@+0.5"]), float(rbad["berr@+0.5"]))
    # stop-gradient directions: forward carries a gradient to P_0 only, backward to P_dt only (through the gather)
    Pg = {0.0: P[0.0].clone().requires_grad_(True), 0.5: P[0.5].clone().requires_grad_(True)}
    rr = cross_time_consistency(Pg, {0.5: st[0.5]}, v_true, grid)
    g_fwd = torch.autograd.grad(rr["fwd@+0.5"], [Pg[0.0], Pg[0.5]], retain_graph=True, allow_unused=True)
    assert g_fwd[0] is not None and g_fwd[0].abs().sum() > 0 and (g_fwd[1] is None or g_fwd[1].abs().sum() == 0)
    g_bwd = torch.autograd.grad(rr["bwd@+0.5"], [Pg[0.0], Pg[0.5]], retain_graph=True, allow_unused=True)
    assert g_bwd[1] is not None and g_bwd[1].abs().sum() > 0 and (g_bwd[0] is None or g_bwd[0].abs().sum() == 0)
    # the motion field gets a gradient (the self-supervised flow signal); cells of weight 0 are ignored
    vg = v_true.clone().requires_grad_(True)
    cross_time_consistency(P, st, vg, grid)["total"].backward()
    assert vg.grad is not None and vg.grad.abs().sum() > 0
    wv = {0.0: torch.ones(1, 1, X, Y), 0.5: torch.zeros(1, 1, X, Y), -0.5: torch.zeros(1, 1, X, Y)}
    assert float(cross_time_consistency(P, st, v_true, grid, weights=wv)["total"]) == 0.0, "no visible cell: no loss"
    print("loss_extras selftest ok: UW-SO weights, group gradient cosines (parallel / opposite / orthogonal), "
          f"cross-time consistency (moving blob {float(r['total']):.3f} vs no motion {float(r0['total']):.3f}; "
          f"pushed-out field {float(rf['total']):.2f} not 0; ego motion exact: matched-cell error "
          f"{float(rst['ferr@+0.5']):.3f} / {float(rst['berr@+0.5']):.3f} (fwd / bwd) vs wrong step "
          f"{float(rbad['ferr@+0.5']):.3f} / {float(rbad['berr@+0.5']):.3f}; stop-gradient per direction; motion field "
          f"gets a gradient)")


if __name__ == "__main__":
    _selftest()
