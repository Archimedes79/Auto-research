"""Grid stage 3: temporal fusion of the sparse map, an ego-motion warp and a learned gate (E2E).

Adapted from bev_fusion/training/temporal_fusion.py (the idea: bring the past into the current frame with the ego motion, read it with ONE bilinear grid_sample,
fuse with a small conv), reduced to what a sparse 5-channel map at 1 m needs: no world-aligned memory, no window of frames, no velocity head (lanes do not move).

state = the previous FUSED sparse map (B, 5, X, Y) [w, cos, sin, ox, oy] + its age (B, 1, X, Y), already in the previous ego frame
step  = pose of the CURRENT ego frame in the PREVIOUS ego frame (B, 3) = (tx, ty, yaw) in metres / radians (x forward, y left)
Hold.forward(cur, state, age, step) -> (fused, age_new); the fused map is the output AND the next state (detach it every N frames for truncated BPTT).

warp: a current cell at p_cur is the previous point p_prev = R(yaw) p_cur + t, read bilinear; the heading channels are rotated with the frame (v_cur = R(-yaw) v_prev),
the offset channels are re-expressed after the sampling grid (offsets are sub-cell, so only rotated, not re-sampled by a whole cell).
fuse: g = sigmoid(conv([cur, warped, valid, age])), fused = g * warped + (1 - g) * cur on the geometry channels, w = 1 - (1 - w_cur)(1 - g w_warp) (noisy-or: a vertex seen twice is more sure).
Run the self-test: python hold.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def warp(x, step, res=1.0, vec=((1, 2),), off=(3, 4)):
    """x (B, C, X, Y) in the previous ego frame -> the current ego frame; returns (warped, valid (B, 1, X, Y)). `vec` index pairs are rotated as vectors, `off` as a vector in cells."""
    B, C, X, Y = x.shape
    tx, ty, yaw = step[:, 0], step[:, 1], step[:, 2]
    cx = (torch.arange(X, device=x.device, dtype=x.dtype) + 0.5 - X / 2) * res            # metres, ego origin in the middle
    cy = (torch.arange(Y, device=x.device, dtype=x.dtype) + 0.5 - Y / 2) * res
    px = cx.view(1, X, 1).expand(B, X, Y)
    py = cy.view(1, 1, Y).expand(B, X, Y)
    c, s = torch.cos(yaw).view(B, 1, 1), torch.sin(yaw).view(B, 1, 1)
    qx = c * px - s * py + tx.view(B, 1, 1)                                               # p_prev = R(yaw) p_cur + t
    qy = s * px + c * py + ty.view(B, 1, 1)
    gx = (qy / res + Y / 2 - 0.5) / max(Y - 1, 1) * 2 - 1
    gy = (qx / res + X / 2 - 0.5) / max(X - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], -1)
    one = torch.ones(B, 1, X, Y, device=x.device, dtype=x.dtype)
    out = F.grid_sample(torch.cat([x, one], 1), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    valid = out[:, -1:]
    out = out[:, :-1] / valid.clamp_min(1e-3)                                             # a mean over the valid taps, edges are not dimmed
    outs = list(out.unbind(1))
    cb, sb = torch.cos(-yaw).view(B, 1, 1), torch.sin(-yaw).view(B, 1, 1)                # v_cur = R(-yaw) v_prev
    for i, j in list(vec) + ([off] if off else []):
        a, b = outs[i], outs[j]
        outs[i], outs[j] = cb * a - sb * b, sb * a + cb * b
    return torch.stack(outs, 1), valid


class Hold(nn.Module):
    def __init__(self, ch=5, hidden=16, res=1.0):
        super().__init__()
        self.res = res
        self.gate = nn.Sequential(nn.Conv2d(2 * ch + 2, hidden, 3, padding=1), nn.ReLU(), nn.Conv2d(hidden, 1, 3, padding=1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -1.0)                                       # start: mostly the current frame, the past is learned in

    def forward(self, cur, state=None, age=None, step=None):
        if state is None:
            return cur, torch.zeros_like(cur[:, :1])
        warped, valid = warp(state, step, self.res)
        age_w, _ = warp(age, step, self.res, vec=(), off=None)
        g = torch.sigmoid(self.gate(torch.cat([cur, warped, valid, age_w / 10.0], 1))) * (valid > 0.5).to(cur.dtype)
        geo = g * warped + (1 - g) * cur
        w = 1 - (1 - cur[:, :1]) * (1 - g * warped[:, :1])                                # noisy-or of the weights
        fused = torch.cat([w, geo[:, 1:]], 1)
        return fused, (age_w + 1.0) * g + (1 - g) * 0.0


def _selftest():
    torch.manual_seed(0)
    X = Y = 64
    res = 1.0

    def observe(tx, ty, yaw):
        """sparse map of a world line (the x-axis of the world, heading (1, 0)) as seen from the ego pose (tx, ty, yaw) in the world (x forward, y left)"""
        cx = (torch.arange(X).float() + 0.5 - X / 2) * res
        cy = (torch.arange(Y).float() + 0.5 - Y / 2) * res
        px = cx.view(X, 1).expand(X, Y)
        py = cy.view(1, Y).expand(X, Y)
        wx = math.cos(yaw) * px - math.sin(yaw) * py + tx
        wy = math.sin(yaw) * px + math.cos(yaw) * py + ty                                  # world point of the cell
        w = torch.exp(-wy * wy / 8.0)                                                      # the line wy = 0, a 4 m wide ridge (bilinear reads of a 1 cell line are not exact)
        hc, hs = math.cos(-yaw), math.sin(-yaw)                                            # world heading (1, 0) in the ego axes
        return torch.stack([w, torch.full((X, Y), hc), torch.full((X, Y), hs), torch.zeros(X, Y), torch.zeros(X, Y)])[None]

    p0 = (0.0, 0.0, 0.0)                         # pose 0 in the world
    p1 = (3.0, 1.0, 0.2)                         # pose 1 in the world
    o0, o1 = observe(*p0), observe(*p1)
    # the pose of frame 1 in frame 0: R(0)^-1 (p1 - p0) = (3, 1), yaw 0.2
    step = torch.tensor([[3.0, 1.0, 0.2]])
    wp, valid = warp(o0, step)
    m = (valid[0, 0] > 0.99) & (o1[0, 0] > 0.5)
    err_w = (wp[0, 0] - o1[0, 0])[m].abs().mean()
    err_h = (wp[0, 1:3] - o1[0, 1:3])[:, m].abs().mean()
    print("warp: weight error", float(err_w), "| heading error", float(err_h))
    assert err_w < 0.1 and err_h < 0.05, "the warp does not reproduce the second observation"
    # identity step is the identity
    ident, v = warp(o0, torch.zeros(1, 3))
    assert (ident - o0)[:, :, 2:-2, 2:-2].abs().max() < 1e-4
    # the module: starts as the current frame, gradient reaches the gate, the noisy-or never exceeds 1
    hold = Hold()
    fused, age = hold(o1, o0, torch.zeros(1, 1, X, Y), step)
    assert (fused[:, 0] - o1[:, 0]).abs().max() < 0.35 and fused[:, 0].max() <= 1.0 + 1e-5
    fused.sum().backward()
    assert hold.gate[-1].weight.grad is not None and hold.gate[-1].weight.grad.abs().sum() > 0   # the zero-started last layer learns first, the first layer after it
    print("hold selftest ok")


if __name__ == "__main__":
    _selftest()
