"""Grid stage 1: the 1 m centre-lane head on the frozen camera + radar BEV features (a small U-Net, stride-2 stem, GroupNorm).

input   x (B, 39, 200, 200): the cached GaussianCaR channels (dec32 PCA + 7 map logits, 0.5 m cells, normalised as `BuilderData`); the old dense head's 3 channels
        (heat, direction) are NOT an input: `x[:, :39]` is sliced here, so the chain does not inherit the old pipeline
output  (B, 6, 100, 100) at 1 m: p logit, cos, sin (raw heading), ox, oy (offset logits, `soft_lines.sparsify` squashes them to +-0.5 cell), s (keep-gate logit)
Axes as the feature raster: rows = x (right), columns = y (forward). The deliverable of stage 2 is `soft_lines.sparsify(head(x))`.
Run the self-test: python head.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

C_IN = 39


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


class GridHead(nn.Module):
    def __init__(self, c_in=C_IN, width=48, keep_bias=2.0, p_bias=-3.0):
        super().__init__()
        w = width
        self.c_in = c_in
        self.stem = block(c_in, w, stride=2)                            # 200 -> 100 (1 m)
        self.e1 = block(w, 2 * w, stride=2)                             # 50
        self.e2 = block(2 * w, 3 * w, stride=2)                         # 25
        self.e3 = block(3 * w, 4 * w, stride=2)                         # 13
        self.d2 = block(4 * w + 3 * w, 3 * w)
        self.d1 = block(3 * w + 2 * w, 2 * w)
        self.d0 = block(2 * w + w, w)
        self.out = nn.Conv2d(w, 6, 1)
        nn.init.normal_(self.out.weight, std=0.01)                       # not zero: with a zero output layer no gradient reaches the body at step 0 (G1)
        with torch.no_grad():
            self.out.bias.copy_(torch.tensor([p_bias, 1.0, 0.0, 0.0, 0.0, keep_bias]))   # few lane cells at the start, heading +x, gate mostly open (the budget closes it)

    def forward(self, x):
        x = x[:, : self.c_in]
        s = self.stem(x)
        a = self.e1(s)
        b = self.e2(a)
        c = self.e3(b)
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)  # noqa: E731
        d = self.d2(torch.cat([up(c, b), b], 1))
        d = self.d1(torch.cat([up(d, a), a], 1))
        d = self.d0(torch.cat([up(d, s), s], 1))
        return self.out(d)


def _selftest():
    torch.manual_seed(0)
    m = GridHead(width=16)
    x = torch.randn(2, 42, 200, 200)                          # 42 channels in: the last three (old dense head) must not matter
    y = m(x)
    assert y.shape == (2, 6, 100, 100)
    x2 = x.clone()
    x2[:, 39:] = torch.randn_like(x2[:, 39:]) * 50
    assert torch.equal(m(x2), y), "the old dense-head channels must not be read"
    n = sum(p.numel() for p in GridHead().parameters())
    print("params", n / 1e6, "M")
    assert 0.5e6 < n < 6e6
    print("head selftest ok")


if __name__ == "__main__":
    _selftest()
