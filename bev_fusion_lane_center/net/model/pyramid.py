"""Stage A of the lane head (E273): the pixel maps, one shared encoder-decoder with three taps (0.5 m lines, 1 m centres + event heats, 2 m road + intersection).

    PyramidHead(width=48)(x (B, 39, 200, 200)) -> dict
        centre1  (B, 12, 100, 100)  E271's six channels (p logit, cos, sin, ox, oy, keep gate; `soft_lines.sparsify` reads them) + six event-heat logits (split, merge, stop, exit, start, end)
        lines05  (B, 6, 200, 200)   logits of lane divider, road divider, road boundary (0-2) and their truncated distances in metres x 1 (3-5, raw: 2 m x sigmoid is applied by the loss)
        road2    (B, 2, 50, 50)     logits of road (drivable area) and intersection
        feat2    (B, 144, 50, 50)   the 2 m feature map of the node grid (decoder level at 2 m + the pooled 1 m level)
        feat1    (B, w, 100, 100)   the 1 m features, feat05 (B, 32, 200, 200) the 0.5 m features (the nodes sample them at their positions)
The body is E271's `GridHead` (same module names), so the weights of a trained E271 head load with `load_e271(path)`; the 1 m tap starts from them.
Axes: rows = x (right), columns = y (forward), 0.5 m cells on the input, origin -50 m.
Run the self-test: python pyramid.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from head import C_IN, block


class PyramidHead(nn.Module):
    def __init__(self, c_in=C_IN, width=48, f05=32):
        super().__init__()
        w = width
        self.c_in, self.w, self.f05 = c_in, w, f05
        self.stem = block(c_in, w, stride=2)                            # 200 -> 100 (1 m)
        self.e1 = block(w, 2 * w, stride=2)                             # 50 (2 m)
        self.e2 = block(2 * w, 3 * w, stride=2)                         # 25
        self.e3 = block(3 * w, 4 * w, stride=2)                         # 13
        self.d2 = block(4 * w + 3 * w, 3 * w)
        self.d1 = block(3 * w + 2 * w, 2 * w)
        self.d0 = block(2 * w + w, w)
        self.centre1 = nn.Conv2d(w, 12, 1)
        self.road2 = nn.Conv2d(2 * w, 2, 1)
        self.s05 = nn.Sequential(nn.Conv2d(c_in, f05, 3, 1, 1, bias=False), nn.GroupNorm(8, f05), nn.ReLU(inplace=True))      # the full-resolution skip
        self.m05 = block(w + f05, f05)
        self.lines05 = nn.Conv2d(f05, 6, 1)
        for m in (self.centre1, self.road2, self.lines05):
            nn.init.normal_(m.weight, std=0.01)
        with torch.no_grad():
            self.centre1.bias.copy_(torch.tensor([-3.0, 1.0, 0.0, 0.0, 0.0, 2.0] + [-4.0] * 6))
            self.road2.bias.copy_(torch.tensor([0.0, -2.0]))
            self.lines05.bias.copy_(torch.tensor([-3.0, -3.0, -3.0, 2.0, 2.0, 2.0]))

    def forward(self, x):
        x = x[:, : self.c_in]
        s = self.stem(x)
        a = self.e1(s)
        b = self.e2(a)
        c = self.e3(b)
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)  # noqa: E731
        d = self.d2(torch.cat([up(c, b), b], 1))
        d1 = self.d1(torch.cat([up(d, a), a], 1))
        d0 = self.d0(torch.cat([up(d1, s), s], 1))
        f05 = self.m05(torch.cat([up(d0, x), self.s05(x)], 1))
        return dict(centre1=self.centre1(d0), lines05=self.lines05(f05), road2=self.road2(d1), feat2=torch.cat([d1, F.avg_pool2d(d0, 2)], 1), feat1=d0, feat05=f05)

    def load_e271(self, path):
        """the body and the six centre channels from a trained E271 head (`runs/e271_a_s<seed>/model.pt`)"""
        sd = torch.load(path, map_location="cpu")["state"]
        own = self.state_dict()
        n = 0
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                n += 1
        own["centre1.weight"][:6] = sd["out.weight"]
        own["centre1.bias"][:6] = sd["out.bias"]
        self.load_state_dict(own)
        return n


def _selftest():
    torch.manual_seed(0)
    m = PyramidHead(width=16, f05=16)
    x = torch.randn(2, 42, 200, 200)
    y = m(x)
    shapes = {k: tuple(v.shape) for k, v in y.items()}
    assert shapes == dict(centre1=(2, 12, 100, 100), lines05=(2, 6, 200, 200), road2=(2, 2, 50, 50), feat2=(2, 48, 50, 50), feat1=(2, 16, 100, 100), feat05=(2, 16, 200, 200)), shapes
    loss = sum(v.square().mean() for v in y.values())
    loss.backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    assert not dead, dead
    n = sum(p.numel() for p in PyramidHead().parameters())
    print("params", n / 1e6, "M")
    print("pyramid selftest ok")


if __name__ == "__main__":
    _selftest()
