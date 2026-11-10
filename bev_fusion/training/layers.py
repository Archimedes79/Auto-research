"""The layer table of the map head (plan 2026-11-03 section 2, "Head spec by kind"): a layer is a ROW (name, kind, level), and the kind
decides the channels, the target and the loss, so a new layer is a row, not new code.

    kind       channels                                                      loss (losses.map_terms)
    area       1 logit: the coverage                                         BCE + region overlap (soft IoU)
    polyline   presence logit [+ truncated distance field, metres, linear]   dilated Dice [+ Huber on the distance] [+ L1 on the unit direction]
               [+ direction (cos, sin), raw, normalised in the loss]
    dense_box  1 logit: the object footprint coverage                        BCE + region overlap
    level      "0.5m": the grid of the map head, the targets are the GT layers.  "1m": one pyramid level down (stride 2): the head reads the
               2 x 2 average-pooled state and its targets are pooled from the 0.5 m GT (pool_targets), no new GT version

Channel order: the rows are walked in table order; a run of consecutive rows of one signature (kind, distance, direction) is a block, written as
all presence channels, then all distance channels, then the direction pairs (the first design's order: the two line masks, then their two
distance fields). `channels(rows)` lists them as (channel name, channel kind, target key), the layout every consumer (model, loss, evaluation,
runner) reads as `spec`. Channel kinds are the consumers' vocabulary: area, line (a polyline with a distance field: scored as a line), centreline
(a polyline without one: the first centre-line design, mask + direction), dist, dir, object. The target key of a 1 m row is the 0.5 m key + "@1m".
The radar aux head's stationary-structure proxy (kind "stationary") is a hook without GT, not a map layer: losses.layer_spec appends it.
Run the self-test: python tests/test_layers.py   (from the bev_fusion folder; CPU, seconds)
"""
import math
import os
import sys
from dataclasses import dataclass
from itertools import groupby

import torch
import torch.nn.functional as F

try:
    from . import launch_saving as LS
except ImportError:                                   # run as a script
    import launch_saving as LS

_BEV_FUSION = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
while not os.path.isdir(os.path.join(_BEV_FUSION, "GtGenerator")) and os.path.dirname(_BEV_FUSION) != _BEV_FUSION:
    _BEV_FUSION = os.path.dirname(_BEV_FUSION)       # a run snapshot (runs/<name>/code/) sits three levels below the bev_fusion folder
if _BEV_FUSION not in sys.path:
    sys.path.append(_BEV_FUSION)                     # the GT layout: the standalone suite GtGenerator/
from GtGenerator.layers import AREAS, DIST_MAX, LINES, OBJECTS  # noqa: E402,F401

KINDS = ("area", "polyline", "dense_box")
LEVELS = {"0.5m": 1, "1m": 2}                         # level -> stride in cells of the 0.5 m GT grid
GT_RES = 0.5                                          # m, the GT grid
EPS = 1e-6


@dataclass(frozen=True)
class Layer:
    name: str                  # the GT layer's name (and the prefix of its channels)
    kind: str                  # area | polyline | dense_box
    level: str = "0.5m"
    dist: bool = False         # polyline: + the truncated distance field (GT `<name>_dist` at 0.5 m; computed from the mask at 1 m)
    direction: bool = False    # polyline: + the direction (cos, sin) (GT `<name>_dir`)

    def __post_init__(self):
        if self.kind not in KINDS or self.level not in LEVELS:
            raise ValueError(f"layer {self.name!r}: kind {self.kind!r} / level {self.level!r} not in {KINDS} / {tuple(LEVELS)}")
        if (self.dist or self.direction) and self.kind != "polyline":
            raise ValueError(f"layer {self.name!r}: a distance field or a direction belongs to a polyline, not to {self.kind!r}")

    def key(self, part="mask"):
        """target key of a part (mask | dist | dir): the GT layer's key at 0.5 m, key@level for a pooled target"""
        base = {"mask": self.name, "dist": self.name + "_dist", "dir": self.name + "_dir"}[part]
        return base if self.level == "0.5m" else f"{base}@{self.level}"


def layer_table(centrelines=False, dense_objects=True, centreline_1m=False):
    """The rows of the map head: 5 areas, 2 line polylines (mask + distance field), [centreline: mask + direction, the first design], 3 dense
    boxes, and at 1 m [the centre-line: presence + distance field + direction]. Old runs keep their layouts: every option is off by default."""
    rows = [Layer(a, "area") for a in AREAS] + [Layer(ln, "polyline", dist=True) for ln in LINES]
    if centrelines:
        rows.append(Layer("centreline", "polyline", direction=True))
    if dense_objects:
        rows += [Layer(o, "dense_box") for o in OBJECTS]
    if centreline_1m:
        rows.append(Layer("centreline", "polyline", "1m", dist=True, direction=True))
    return rows


def _mask_kind(r):
    return {"area": "area", "dense_box": "object"}.get(r.kind) or ("line" if r.dist else "centreline")


def channels(rows, level="0.5m"):
    """[(channel name, channel kind, target key)] of the rows of one level, in the order of the block rule above"""
    out = []
    for _, block in groupby([r for r in rows if r.level == level], key=lambda r: (r.kind, r.dist, r.direction)):
        block = list(block)
        out += [(r.name, _mask_kind(r), r.key()) for r in block]
        out += [(r.name + "_dist", "dist", r.key("dist")) for r in block if r.dist]
        for r in block:
            if r.direction:
                out += [(r.name + "_cos", "dir", r.key("dir")), (r.name + "_sin", "dir", r.key("dir"))]
    return out


def source_keys(rows):
    """the GT layers (GtGenerator keys) the targets of the rows are made from: a 0.5 m row's own keys, a pooled row's 0.5 m mask and direction"""
    keys = []
    for r in rows:
        own = ([r.name] + ([r.name + "_dist"] if r.dist and r.level == "0.5m" else []) + ([r.name + "_dir"] if r.direction else []))
        keys += [k for k in own if k not in keys]
    return keys


# ------------------------------------------------------------------------------------------------ targets of a pooled level
def truncated_distance(mask, res=GT_RES, dist_max=DIST_MAX):
    """(B, 1, X, Y) mask {0, 1} -> (B, 1, X, Y) metres: from each cell centre to the nearest centre of a mask cell, capped at dist_max. Exact:
    one convolution with the discs of every distinct radius below dist_max says whether a mask cell lies within it (the cap is 2 m = 4 cells)."""
    if LS.on("sync"):                                       # D53: the discs and radii are constants: built once per (res, dist_max, device), no mask index / device copy per call
        R, radii, discs, full = _discs(res, dist_max, mask.device)
        hit = F.conv2d(mask.float(), discs, padding=R) > 0.5
        return torch.where(hit, radii.view(1, -1, 1, 1), full).amin(1, keepdim=True)
    R = int(math.ceil(dist_max / res - 1e-9))
    off = torch.arange(-R, R + 1, dtype=torch.float32)
    rr = torch.hypot(off[:, None], off[None, :]) * res
    radii = torch.unique(torch.round(rr[rr < dist_max - 1e-6] * 1e6) / 1e6)                  # sorted, starts at 0
    discs = (rr[None] <= radii.view(-1, 1, 1) + 1e-6).float().unsqueeze(1).to(mask.device)    # (n, 1, 2R+1, 2R+1)
    hit = F.conv2d(mask.float(), discs, padding=R) > 0.5                                     # (B, n, X, Y): a mask cell within radii[k]
    d = torch.where(hit, radii.view(1, -1, 1, 1).to(mask.device), torch.full((), float(dist_max), device=mask.device))
    return d.amin(1, keepdim=True)


_DISC_CACHE = {}


def _discs(res, dist_max, device):
    """(R, radii, discs, cap) of truncated_distance on `device`, the very tensors the plain code builds on every call, kept"""
    key = (res, dist_max, str(device))
    if key not in _DISC_CACHE:
        R = int(math.ceil(dist_max / res - 1e-9))
        off = torch.arange(-R, R + 1, dtype=torch.float32)
        rr = torch.hypot(off[:, None], off[None, :]) * res
        radii = torch.unique(torch.round(rr[rr < dist_max - 1e-6] * 1e6) / 1e6)
        discs = (rr[None] <= radii.view(-1, 1, 1) + 1e-6).float().unsqueeze(1).to(device)
        _DISC_CACHE[key] = (R, radii.to(device), discs, torch.full((), float(dist_max), device=device))
    return _DISC_CACHE[key]


def pool_polyline(mask, direction=None, stride=2, res=GT_RES, dist_max=DIST_MAX):
    """0.5 m GT of a polyline -> its targets at a coarser level (stride cells): (mask, dist, direction), shapes (B, 1 / 1 / 2, X / s, Y / s).
    mask: max-pool (a block holds the line if any of its cells does); dist: the truncated distance field (metres) of the 0.5 m mask, min-pooled
    (the line cell nearest to the block, so it is 0 exactly where the mask is 1); direction: the mask-weighted mean of the block's (cos, sin),
    0 where the block has no line cell (not renormalised: two lanes of different heading average to a shorter vector, as in the 0.5 m GT; the
    loss compares the unit prediction with it). direction None -> (mask, dist, None)."""
    X, Y = mask.shape[-2:]
    if X % stride or Y % stride:
        raise ValueError(f"a {X} x {Y} grid does not pool by {stride}")
    m = F.max_pool2d(mask, stride)
    d = -F.max_pool2d(-truncated_distance(mask, res, dist_max), stride)
    if direction is None:
        return m, d, None
    n = F.avg_pool2d(mask, stride)
    u = F.avg_pool2d(mask * direction, stride) / n.clamp(min=EPS)                             # sum / sum: the 1 / s^2 cancels
    return m, d, torch.where(n > 0, u, torch.zeros_like(u))


def pool_targets(rows, T):
    """The targets and the line weights of every pooled row (level != 0.5m) from the 0.5 m ones in T -> {key: tensor}: T[r.key(part)] of each
    part the row has, and `w_lines@level` (the mean of the 0.5 m weight over the block)."""
    out = {}
    for r in rows:
        if r.level == "0.5m":
            continue
        if r.kind != "polyline":
            raise NotImplementedError(f"layer {r.name!r}: only polylines are pooled to {r.level} so far")
        s = LEVELS[r.level]
        m, d, u = pool_polyline(T[r.name], T[r.name + "_dir"] if r.direction else None, s)
        out[r.key()] = m
        if r.dist:
            out[r.key("dist")] = d
        if r.direction:
            out[r.key("dir")] = u
        out[f"w_lines@{r.level}"] = F.avg_pool2d(T["w_lines"], s)
    return out
