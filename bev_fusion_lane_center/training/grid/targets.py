"""Training targets of the grid head from the ground-truth lane points: what the head must output in every 1 m cell (shared by the losses and the oracle test).

Frame: the feature raster of the project, rows = x (right), columns = y (forward), 0.5 m cells on the input (200 x 200), 1 m cells on the head (100 x 100), origin -50 m;
GT lane points as `p16_data.gt_tensors`: g05 (B, G, 2) metres every 0.5 m, padded with 1e4, t05 (B, G, 2) the travel tangent.

cell_targets -> d (B, n, n) distance of the cell centre to the nearest lane point (m), off (B, 2, n, n) nearest point minus cell centre in cells,
                head (B, 2, n, n) tangent of that nearest point (unit), idx (B, n, n) index of that point.
heat(d, sigma) -> soft centre-lane target exp(-d^2 / 2 sigma^2).
Run the self-test: python targets.py
"""
import torch


def centres(n=100, res=1.0, origin=-50.0, device="cpu"):
    c = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) * res + origin
    return torch.stack([c.view(n, 1).expand(n, n), c.view(1, n).expand(n, n)], -1).reshape(n * n, 2)


def cell_targets(g05, t05, n=100, res=1.0, origin=-50.0):
    B = g05.shape[0]
    c = centres(n, res, origin, g05.device)
    d, off, head, idx = [], [], [], []
    for b in range(B):                                                   # one frame at a time: (n * n, G) floats, not (B, n * n, G)
        dist = torch.cdist(c, g05[b])                                    # padded points (1e4) are never the nearest
        dm, im = dist.min(1)
        d.append(dm.view(n, n))
        off.append(((g05[b][im] - c) / res).T.reshape(2, n, n))
        head.append(t05[b][im].T.reshape(2, n, n))
        idx.append(im.view(n, n))
    return torch.stack(d), torch.stack(off), torch.stack(head), torch.stack(idx)


def heat(d, sigma=1.0):
    return torch.exp(-(d * d) / (2 * sigma * sigma))


def _selftest():
    g = torch.stack([torch.zeros(120), torch.linspace(-30, 29.5, 120)], 1)[None]        # a lane along y at x = 0, heading +y
    g = torch.cat([g, torch.full((1, 10, 2), 1e4)], 1)
    t = torch.cat([torch.tensor([0.0, 1.0]).expand(120, 2), torch.zeros(10, 2)], 0)[None]
    d, off, head, idx = cell_targets(g, t)
    # cell (49, 70) has its centre at x = -0.5, y = 20.5: the nearest lane point is (0, 20.5) -> offset (+0.5, 0) cells, distance 0.5
    assert abs(float(d[0, 49, 70]) - 0.5) < 1e-4 and abs(float(off[0, 0, 49, 70]) - 0.5) < 1e-4 and abs(float(off[0, 1, 49, 70])) < 0.26, (d[0, 49, 70], off[0, :, 49, 70])
    assert torch.allclose(head[0, :, 49, 70], torch.tensor([0.0, 1.0]))
    assert float(d[0, 10, 10]) > 20 and float(heat(d)[0, 10, 10]) < 1e-6                 # far from the lane
    assert float(heat(d)[0, 49, 70]) > 0.8
    print("targets selftest ok")


if __name__ == "__main__":
    _selftest()
