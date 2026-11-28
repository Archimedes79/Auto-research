"""Loss pieces that are not part of BEVLoss (notes in literature/papers): a diagnostic of the task groups and the cross-time consistency of the
predictions (net/chain.py and net/loss/train_terms.py use them; losses.py imports the weighted-sum helpers `_wsum` / `_wcount` from here). Conventions as
losses.py: prob / target (B, K, X, Y), weights w broadcastable (a cell
of weight 0 is ignored), per-channel results (K,), 0 for a channel without GT.

1  group_gradient_cosines  diagnostic: cosine and norm of the task-group gradients at the shared BEV feature (head
                           backward passes only) - look at it before any gradient surgery.
2  cross_time_consistency  train-only cross-time consistency of the predictions at dt = 0 and dt = +-0.5 s (MapTCL
                           2608.05209, ST-P3 / FIERY-style warping, UnFlow 1711.07837 masks): the prediction at dt = 0 is
                           carried to the other time by the splat geometry (ego steps + the motion field) and compared,
                           BOTH directions (forward: splat P_0 and compare with P_b, stop-grad on P_b; backward: gather
                           P_b at the carried positions and compare with P_0, stop-grad on P_0), only where both sides are
                           visible, cells without a counterpart cost a constant (no trivial solution by pushing everything
                           out of the grid), age weights. Nothing at inference.
Tests: tests/test_loss_extras.py
"""
import math

import torch
import torch.nn.functional as F

from net.model.splat import chain_positions, grid_points, splat

EPS = 1e-6


def _wsum(x, w):
    return (x * w).sum((0, 2, 3))


def _wcount(w, like):
    return w.expand_as(like).sum((0, 2, 3)).clamp(min=EPS)


# ------------------------------------------------------------------------------------------------ 1
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


# ------------------------------------------------------------------------------------------------ 2
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
