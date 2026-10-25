"""CNT-Net: moderner dichter Decoder fuer die BEV-Merkmale (Abstand zur Mittellinie, 200 x 200 bei 0,5 m).

Bausteine nach dem Stand der Technik (ConvNeXt V2, Transformer-Flaschenhals wie in BEV-Segmentierungsnetzen):
  - Eingang: 1x1-Reduktion der Merkmale (wie HeatNet/UNetHeat), Umsampling der 128 x 128-Merkmale (0,8 m) auf 200 x 200 (0,5 m)
  - Encoder: ConvNeXt-V2-Bloecke (7x7-Tiefenfaltung, LayerNorm, Punktfaltungen, GELU, GRN) auf 200 -> 100 -> 50 Zellen
  - Flaschenhals: Transformer-Encoder (globale Selbstaufmerksamkeit, gelernte Positionscodierung, Pre-LN) auf 25 x 25 = 625 Token
  - Decoder: U-Netz-artige Hochskalierung mit Skip-Verbindungen, Ausgabe 1 Kanal (Abstand, wie bisher sigmoid(logit) = Abstand/8 Pixel)
Gleiche Ein-/Ausgabe wie HeatNet/UNetHeat: forward(feats, sem, full=False).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

HALF = 50.0


class LN2d(nn.Module):
    """LayerNorm ueber die Kanaele fuer NCHW-Tensoren."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(c))
        self.b = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        m = xf.mean(1, keepdim=True)
        v = (xf - m).pow(2).mean(1, keepdim=True)
        y = (xf - m) / torch.sqrt(v + self.eps)
        return (y * self.w[None, :, None, None] + self.b[None, :, None, None]).to(x.dtype)


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2)."""

    def __init__(self, c):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        gx = torch.norm(x.float(), p=2, dim=(2, 3), keepdim=True)
        nx = (gx / (gx.mean(dim=1, keepdim=True) + 1e-6)).to(x.dtype)
        return self.g * (x * nx) + self.b + x


class CNXBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c)
        self.ln = LN2d(c)
        self.pw1 = nn.Conv2d(c, 4 * c, 1)
        self.act = nn.GELU()
        self.grn = GRN(4 * c)
        self.pw2 = nn.Conv2d(4 * c, c, 1)

    def forward(self, x):
        return x + self.pw2(self.grn(self.act(self.pw1(self.ln(self.dw(x))))))


def stage(c, n):
    return nn.Sequential(*[CNXBlock(c) for _ in range(n)])


def down(ci, co):
    return nn.Sequential(LN2d(ci), nn.Conv2d(ci, co, 2, stride=2))


def dec_block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(), nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU())


class CNTNet(nn.Module):
    def __init__(self, cin=32, c=32, base=64, drop=0.2, depths=(2, 2, 3), tdepth=4, heads=8, use_feat=True, use_sem=False, out_ch=1):
        super().__init__()
        assert use_feat and not use_sem, "CNTNet nutzt nur die BEV-Merkmale (kein sem-Eingang)"
        b = base  # Kanalbreiten je Aufloesung: 200 -> b/2, 100 -> b, 50 -> 2b, 25 -> 4b (Transformer)
        self.red = nn.Sequential(nn.Conv2d(cin, c, 1), nn.GroupNorm(8, c), nn.ReLU())
        self.drop = nn.Dropout2d(drop)
        v = (-HALF + (torch.arange(200).float() + 0.5) * 0.5) / 51.2  # Zellmitten, normiert auf die Merkmalsausdehnung (+-51,2 m)
        self.register_buffer("grid", torch.stack([v[None, :].expand(200, 200), v[:, None].expand(200, 200)], -1)[None])
        self.stem = nn.Sequential(nn.Conv2d(c, b // 2, 3, padding=1), nn.GroupNorm(8, b // 2), nn.GELU(), nn.Conv2d(b // 2, b // 2, 3, padding=1), nn.GroupNorm(8, b // 2), nn.GELU())
        self.d1, self.s2, self.d2, self.s3, self.d3 = down(b // 2, b), stage(b, depths[1]), down(b, 2 * b), stage(2 * b, depths[2]), down(2 * b, 4 * b)
        self.pos = nn.Parameter(torch.zeros(1, 25 * 25, 4 * b))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(4 * b, heads, 16 * b, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.trans = nn.TransformerEncoder(layer, tdepth)
        self.u3, self.u2, self.u1 = dec_block(4 * b + 2 * b, 2 * b), dec_block(2 * b + b, b), dec_block(b + b // 2, b // 2)
        self.out = nn.Conv2d(b // 2, out_ch, 1)

    def forward(self, feats, sem=None, full=False):
        x = self.red(feats.float())
        x = F.grid_sample(x, self.grid.expand(x.shape[0], -1, -1, -1), mode="bilinear", align_corners=False)
        e1 = self.stem(self.drop(x))  # 200
        e2 = self.s2(self.d1(e1))  # 100
        e3 = self.s3(self.d2(e2))  # 50
        t = self.d3(e3)  # 25
        B, C, H, W = t.shape
        t = self.trans(t.flatten(2).transpose(1, 2) + self.pos.to(t.dtype)).transpose(1, 2).reshape(B, C, H, W)
        up = lambda z: F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False)
        u3 = self.u3(torch.cat([up(t), e3], 1))
        u2 = self.u2(torch.cat([up(u3), e2], 1))
        u1 = self.u1(torch.cat([up(u2), e1], 1))
        o = self.out(u1)
        return o if full else o[:, 0]


if __name__ == "__main__":
    net = CNTNet()
    print("Parameter:", sum(p.numel() for p in net.parameters()) / 1e6, "Mio.")
    x = torch.randn(2, 32, 128, 128)
    print(net(x, None).shape)
