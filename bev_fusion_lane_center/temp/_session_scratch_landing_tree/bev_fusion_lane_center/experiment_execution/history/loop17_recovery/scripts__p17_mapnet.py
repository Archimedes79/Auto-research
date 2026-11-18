"""Loop 17 / E138: a map-conditioned builder. The SD-map raster (5 channels of sd_lib.raster_sd: route, roads, road direction x / y, lane count) is appended to the 42 cached BEV channels; only the first conv of the stem is widened
(new input weights start at zero, so the widened network starts as the champion). Training (p17_mapft.py) shows the network three map situations (as E87: a map with a scene-constant pose error sigma in {0, 1, 2, 3} m, a wrong map 10 %, no map 10 %)
and loses perception on purpose in 40 % of the frames (global attenuation of heat / direction / features, or rectangles without any evidence) so that the map has something to add; the real failure series are never seen in training.
Usage: imported by p17_mapft.py / p17_eval.py."""
import os
import sys
import zlib

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p11_route_time as RT11  # noqa: E402
import p8_lib as L  # noqa: E402
import sd_lib as SL  # noqa: E402

N_SD = 5


def widen(net):
    """first conv of the stem: 42 -> 42 + 5 input channels, the new ones start at zero"""
    first = next((n, m) for n, m in net.stem.c0.named_modules() if isinstance(m, nn.Conv2d))
    name, conv = first
    new = nn.Conv2d(conv.in_channels + N_SD, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None).to(conv.weight.device)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : conv.in_channels] = conv.weight
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    parent = net.stem.c0
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    if parts[-1].isdigit():
        parent[int(parts[-1])] = new
    else:
        setattr(parent, parts[-1], new)
    return net


def degrade(x, rng):
    """loss of perception on purpose: global attenuation of the evidence, or rectangles without any evidence (x: [42, H, W] float, standardised channels)"""
    x = x.clone()
    r = rng.rand()
    if r < 0.5:
        s = rng.uniform(0.0, 0.4)
        x[39] *= s
        x[40:42] *= s + 0.2
        x[:39] = x[:39] * rng.uniform(0.3, 0.8) + torch.from_numpy(rng.randn(39, 200, 200).astype(np.float32)) * 0.5
    else:
        for _ in range(int(rng.randint(1, 4))):
            h, w = int(rng.randint(60, 121)), int(rng.randint(60, 121))
            i, j = int(rng.randint(0, 200 - h)), int(rng.randint(0, 200 - w))
            x[:, i : i + h, j : j + w] = 0.0
    return x


class MapWrap(torch.utils.data.Dataset):
    """items of a builder dataset with the SD raster appended to x. train=True: random map situation + perception loss; otherwise one fixed situation (mode exact with pose error sigma / none / wrong)."""

    def __init__(self, base, part, P, train=False, mode="exact", sigma=0.0, p_none=0.1, p_wrong=0.1, p_degrade=0.4, sigmas=(0.0, 1.0, 2.0, 3.0), seed=0):
        self.base, self.part, self.P, self.train = base, part, P, train
        self.mode, self.sigma, self.p_none, self.p_wrong, self.p_degrade, self.sigmas, self.seed = mode, sigma, p_none, p_wrong, p_degrade, sigmas, seed
        self.sdl = None
        self.gt = getattr(base, "gt", None)
        self.rows = getattr(base, "rows", None)
        self.ks = getattr(base, "ks", None)

    def __len__(self):
        return len(self.base)

    def __getattr__(self, k):  # predict() / evaluation read gt, rows ... of the dataset
        if k in ("base", "sdl"):
            raise AttributeError(k)
        return getattr(self.base, k)

    def raster(self, row, rng, mode, sigma):
        if mode == "none":
            return np.zeros((N_SD, 200, 200), np.float32)
        if mode == "wrong":
            for _ in range(20):
                r2 = int(rng.randint(len(self.sdl)))
                if r2 != row and self.sdl[r2] is not None and int(self.P["scene"][r2]) != int(self.P["scene"][row]):
                    sdm, _, _ = L.draw_sd(self.sdl[r2], r2, 0.0, 0.0, 1)
                    return SL.raster_sd(sdm).astype(np.float32)
            return np.zeros((N_SD, 200, 200), np.float32)
        sd = self.sdl[row]
        if sd is None:
            return np.zeros((N_SD, 200, 200), np.float32)
        if sigma == 0:
            sdm, _, _ = L.draw_sd(sd, row, 0.0, 0.0, 1)
            return SL.raster_sd(sdm).astype(np.float32)
        return RT11.sd_raster(sd, row, int(self.P["scene"][row]), sigma, self.P, False).astype(np.float32)

    def __getitem__(self, i):
        if self.sdl is None:
            self.sdl = L.cache("sd", self.part)
        o = self.base[i]
        row = int(o["row"])
        x = o["x"].float()
        if self.train:
            rng = np.random.RandomState((self.seed * 7919 + zlib.crc32(f"{row}:{i}".encode())) % (2 ** 32))
            u = rng.rand()
            mode = "none" if u < self.p_none else ("wrong" if u < self.p_none + self.p_wrong else "exact")
            sigma = float(self.sigmas[int(rng.randint(len(self.sigmas)))])
            if rng.rand() < self.p_degrade:
                x = degrade(x, rng)
        else:
            rng = np.random.RandomState((zlib.crc32(f"{row}:{self.mode}".encode()) + self.seed) % (2 ** 32))
            mode, sigma = self.mode, self.sigma
        o["x"] = torch.cat([x, torch.from_numpy(self.raster(row, rng, mode, sigma))], 0).half()
        return o
