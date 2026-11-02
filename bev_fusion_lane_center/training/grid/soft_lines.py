"""Grid stage 1 + 2: the 1 m centre-lane map and its differentiable sparsification (E2E, no argmax, no rule step).

Conventions as the BEV system (bev_fusion/training/extract.py): arrays (B, C, X, Y), x forward (rows), y left (columns), heading as (cos, sin) in (x, y).

head output per cell (6 channels, `HEAD_CH`): p (centre-lane logit), cos, sin (heading, raw), ox, oy (sub-cell offset of the line in cells, tanh * 0.5), s (keep-gate logit)

sparsify(out) -> sparse map (B, 5, X, Y) = [w, cos, sin, ox, oy]
    w = sigmoid(p) * ridge * keep   ridge: soft maximum across the line (the two neighbours along the normal, bilinear), so a 3 m wide ridge becomes one cell wide;
                                    keep : sigmoid(s), the learned gate that lets a cell stay as a vertex (trained under the vertex budget `budget_loss`)
    no top-k and no threshold on the training path: w is a weight per cell, a vertex is a cell with weight; the delivered list is a read-out (`vertices`) at inference only.

Copied from bev_fusion/training/extract.py: `soft_line_map` (distance field -> line probability, used for the loss against the GT line cells). The rest is new and small.
Run the self-test: python soft_lines.py
"""
import math

import torch
import torch.nn.functional as F

HEAD_CH = ("p", "cos", "sin", "ox", "oy", "s")


def soft_line_map(dist, res, k=None):
    """(copy of bev_fusion extract.soft_line_map) distance field (..., X, Y) in metres -> line probability (..., X, Y); k: the sharpness (default 50 / res)"""
    k = 50.0 / res if k is None else k
    d = dist
    pad = F.pad(d.reshape(-1, 1, *d.shape[-2:]), (1, 1, 1, 1), value=float("inf")).reshape(*d.shape[:-2], d.shape[-2] + 2, d.shape[-1] + 2)
    nx = torch.minimum(pad[..., :-2, 1:-1], pad[..., 2:, 1:-1])
    ny = torch.minimum(pad[..., 1:-1, :-2], pad[..., 1:-1, 2:])
    near = torch.sigmoid(k * (res / 2 + 1e-9 - d))
    loc = torch.maximum(torch.sigmoid(k * (nx - d)), torch.sigmoid(k * (ny - d)))
    return torch.minimum(near, loc)


def _shift(p, dx, dy):
    """p (B, 1, X, Y) sampled at (x + dx, y + dy) (cells, tensors (B, 1, X, Y)), bilinear, zeros outside"""
    B, _, X, Y = p.shape
    ix = torch.arange(X, device=p.device, dtype=p.dtype).view(1, 1, X, 1).expand(B, 1, X, Y)
    iy = torch.arange(Y, device=p.device, dtype=p.dtype).view(1, 1, 1, Y).expand(B, 1, X, Y)
    gx = (iy + dy) / max(Y - 1, 1) * 2 - 1       # grid_sample: [..., 0] indexes the width (y), [..., 1] the height (x)
    gy = (ix + dx) / max(X - 1, 1) * 2 - 1
    return F.grid_sample(p, torch.stack([gx[:, 0], gy[:, 0]], -1), mode="bilinear", padding_mode="zeros", align_corners=True)


def ridge(prob, cos, sin, k=20.0):
    """soft 'is the maximum across the line': compare p with the two neighbours along the normal n = (-sin, cos); 1 on the crest, ~0 beside it"""
    n = torch.sqrt(cos * cos + sin * sin + 1e-6)
    nx, ny = -sin / n, cos / n
    a = _shift(prob, nx, ny)
    b = _shift(prob, -nx, -ny)
    return torch.sigmoid(k * (prob - a + 1e-3)) * torch.sigmoid(k * (prob - b + 1e-3))


def sparsify(out, k=20.0):
    """head output (B, 6, X, Y) -> sparse map (B, 5, X, Y) [w, cos, sin, ox, oy] (unit heading); differentiable in every channel"""
    prob = torch.sigmoid(out[:, 0:1])
    h = out[:, 1:3]
    h = h / torch.sqrt((h * h).sum(1, keepdim=True) + 1e-6)
    off = torch.tanh(out[:, 3:5]) * 0.5
    w = prob * ridge(prob, h[:, 0:1], h[:, 1:2], k) * torch.sigmoid(out[:, 5:6])
    return torch.cat([w, h, off], 1)


def budget(sparse, lines_m, per_100m=11.0):
    """vertex budget (the prototype delivers 10.9 node ids per 100 m): the excess of the vertex mass over the budget, per frame,
    relu(sum w - per_100m * lines_m / 100) / (per_100m * lines_m / 100 + 1); lines_m (B,) = GT line length in metres. Only the excess is punished,
    so the recall loss can still ask for vertices where the line bends"""
    target = per_100m * lines_m / 100.0
    return F.relu(sparse[:, 0].sum((1, 2)) - target) / (target + 1.0)


def positions(sparse, res=1.0, origin=(-50.0, -50.0)):
    """metric position (x, y) of every cell's vertex (B, 2, X, Y) = cell centre + sub-cell offset; origin = (x, y) of the corner of cell (0, 0)"""
    B, _, X, Y = sparse.shape
    x = (torch.arange(X, device=sparse.device, dtype=sparse.dtype) + 0.5) * res + origin[0]
    y = (torch.arange(Y, device=sparse.device, dtype=sparse.dtype) + 0.5) * res + origin[1]
    cx = x.view(1, X, 1).expand(B, X, Y)
    cy = y.view(1, 1, Y).expand(B, X, Y)
    return torch.stack([cx + sparse[:, 3] * res, cy + sparse[:, 4] * res], 1)


def vertices(sparse, res=1.0, origin=(-50.0, -50.0), thr=0.3):
    """READ-OUT at inference (not on the training path): the cells with w >= thr as arrays pos (N, 2), heading (N, 2), w (N,), per frame in a list"""
    pos = positions(sparse, res, origin)
    out = []
    for b in range(sparse.shape[0]):
        m = sparse[b, 0] >= thr
        out.append((pos[b][:, m].T, sparse[b, 1:3][:, m].T, sparse[b, 0][m]))
    return out


def _selftest():
    torch.manual_seed(0)
    X = Y = 64
    ix = torch.arange(X).view(X, 1).float()
    iy = torch.arange(Y).view(1, Y).float()
    # a diagonal line (heading 45 deg) 3 cells wide
    d = ((ix - iy) / math.sqrt(2)).abs()
    prob = torch.exp(-d * d / 2.0).view(1, 1, X, Y)
    logit = torch.log(prob.clamp(1e-4, 1 - 1e-4) / (1 - prob.clamp(1e-4, 1 - 1e-4)))
    c = math.cos(math.pi / 4)
    out = torch.cat([logit, torch.full((1, 1, X, Y), c), torch.full((1, 1, X, Y), c), torch.zeros(1, 2, X, Y), torch.full((1, 1, X, Y), 4.0)], 1)
    out.requires_grad_(True)
    sp = sparsify(out)
    w = sp[0, 0]
    crest = (ix - iy).abs() < 0.5
    side = ((ix - iy).abs() > 1.9) & ((ix - iy).abs() < 4.1)
    print("w on the crest", float(w[crest].mean()), "| beside it", float(w[side].mean()))
    assert w[crest][5:-5].mean() > 0.8 and w[side].mean() < 0.15, "the ridge must be thin"
    # the same ridge, but the keep gate off -> no vertices
    out2 = out.detach().clone()
    out2[:, 5] = -6.0
    assert sparsify(out2)[0, 0].max() < 0.01
    # gradient reaches every channel
    (sp[:, 0].sum() + sp[:, 1:].sum() * 0.1).backward()
    g = out.grad.abs().sum((0, 2, 3))
    print("gradient per channel", [f"{v:.2e}" for v in g.tolist()])
    assert (g > 0).all(), "a channel without gradient"
    # sub-cell offset moves the position
    pos = positions(sp.detach())
    assert pos.shape == (1, 2, X, Y)
    # budget: a map with many vertices is punished, a sparse one is not
    assert float(budget(sp.detach(), torch.tensor([100.0]))) > 0
    v = vertices(sp.detach())[0]
    assert len(v[2]) > 0
    # soft_line_map copy: distance field of a straight line gives a one-cell line
    dist = (iy - 32).abs().expand(X, Y).clone().view(1, X, Y)
    lm = soft_line_map(dist, 1.0)
    assert lm[0, :, 32].min() > 0.9 and lm[0, :, 30].max() < 0.1
    print("soft_lines selftest ok")


if __name__ == "__main__":
    _selftest()
