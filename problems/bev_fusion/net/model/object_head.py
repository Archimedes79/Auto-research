"""Object head, stage 1 (S5a): CenterPoint-style centre points on the BEV grid.

Pixel-based: every cell is classified as an object centre per class group (heatmap); at the centre cell the head
regresses the sub-cell offset, the velocity and, by kind of group, an oriented box (log length, log width, yaw as
sin, cos) or a circle (log radius). Decoding: local maxima of the heatmap (3 x 3 max-pool), top-K, score threshold.
Training targets: Gaussian peaks around the GT centres with CenterPoint's radius rule (min overlap 0.1, at least 2
cells); losses: focal loss on the heatmap (CenterNet), L1 on the regressions at the GT centre cells.
Basis: CenterNet "Objects as Points" (1904.07850, from memory), CenterPoint 2006.11275 (lit; MIT code
`det3d/models/bbox_heads/center_head.py` - re-implemented here, nothing copied).
Boxes are dicts: cx, cy (m, vehicle frame), length, width (m), yaw (rad), radius (m, circles), vx, vy (m/s), group.
Tests: tests/test_object_head.py
"""
import math


import torch
import torch.nn as nn


GROUPS = (("car", "box"), ("truck", "box"), ("pedestrian", "circle"))     # (name, kind); synthetic-world classes


def _branch(cin, hidden, cout, bias=0.0):
    b = nn.Sequential(nn.Conv2d(cin, hidden, 3, padding=1), nn.ReLU(), nn.Conv2d(hidden, cout, 1))
    nn.init.constant_(b[2].bias, bias)
    return b


DIR_OFFSET = -math.pi / 4      # direction classes split the heading circle at -45 deg / 135 deg (diagonal: rare on roads)
MOVING_SPEED = 0.5             # m/s: target of the mov logit, = evaluation/metrics.py MOVING_SPEED (the AVE's moving / static split)


def direction_class(yaw):
    """1 if the heading lies in [DIR_OFFSET, DIR_OFFSET + pi) (mod 2 pi), else 0 (SECOND's direction target)."""
    return float(((yaw - DIR_OFFSET) % (2 * math.pi)) < math.pi)


class ObjectHead(nn.Module):
    """BEV features (B, C, X, Y) -> dict of maps: hm (B, G, X, Y) logits, off (2), size (2: log l, log w),
    rot (2: sin, cos), rad (1: log r), vel (2). direction=True (t1 round 1, 2026-10-26): also dir (1 logit); the rot
    loss is then sign-free (the box axis) and the front / back half comes from dir, as SECOND (Yan et al. 2018) /
    PointPillars: the plain L1 on (sin, cos) did not learn front vs back on nuScenes in 2000 steps (rot loss stayed at
    the level of random directions while the axis was right). moving=True (t11 hurdle, 2026-11-01): also mov (1 logit,
    moving / static); the velocity is then regressed on moving centres only and decode zeroes it where mov says static:
    the L1 over all centres, 64 % of them static, had its minimiser (the median) at 0 m/s."""

    def __init__(self, cin, groups=GROUPS, hidden=64, direction=False, moving=False):
        super().__init__()
        self.groups = groups
        self.shared = nn.Sequential(nn.Conv2d(cin, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU())
        self.hm = _branch(hidden, hidden, len(groups), bias=-2.19)        # sigmoid(-2.19) = 0.1 (CenterNet init)
        self.off, self.size = _branch(hidden, hidden, 2), _branch(hidden, hidden, 2)
        self.rot, self.rad, self.vel = _branch(hidden, hidden, 2), _branch(hidden, hidden, 1), _branch(hidden, hidden, 2)
        self.names = ("hm", "off", "size", "rot", "rad", "vel") + (("dir",) if direction else ()) + (("mov",) if moving else ())
        if direction:
            self.dir = _branch(hidden, hidden, 1)
        if moving:
            self.mov = _branch(hidden, hidden, 1, bias=-1.39)             # sigmoid(-1.39) = 0.2: prior share of moving centres

    def forward(self, feat):
        x = self.shared(feat)
        return {k: getattr(self, k)(x) for k in self.names}


# ------------------------------------------------------------------------------------------------ decoding
_boxes_from_maps = None


@torch.no_grad()
def decode(out, grid, groups=GROUPS, k=50, threshold=0.2, from_probabilities=False):
    """out maps -> list (B) of box dicts with 'score': the local maxima of the heatmap (3 x 3 max-pool), top-k, score threshold. With
    out['mov'] the velocity is 0 where sigmoid(mov) < 0.5 (static: the AVE-optimal answer of a two-point mixture; inference only).
    There is ONE extractor, extract.boxes_from_maps (vectorised, only the K peak cells are read); this is its name in the object head's
    module. extract.py's self-test checks it box by box against the plain per-box loop. from_probabilities: hm already in [0, 1]."""
    global _boxes_from_maps
    if _boxes_from_maps is None:                           # imported on first use: extract imports this module (DIR_OFFSET)
        from bev_fusion.net.model.extract import boxes_from_maps
        _boxes_from_maps = boxes_from_maps
    return _boxes_from_maps(out, grid, groups, k, threshold, from_probabilities)
