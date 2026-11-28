"""Radar block (simple, the owner 2026-10-25): every detection goes into the BEV cell where it resides, together with its
properties, followed by two 3x3 convolutions. No point network, no learned footprint, no polar image (those stay later variants,
see bev_fusion/radar_encoding.md).

Per detection (all supplied by an adapter, in vehicle coordinates, so any rig works):
  x, y   position in the vehicle frame at the OUTPUT time (metres)       r    distance from the sensor (metres)
  psi    direction from the sensor to the detection in the vehicle frame at the time of the measurement (radians): the
         ORIGINAL ray direction - after an ego-motion shift of an older sweep it is not recomputed from x, y
  vr     radial (Doppler) velocity along that ray (m/s); the adapter decides what it is (ego-compensated for nuScenes)
  rcs    radar cross section (dBsm)                                      dt   age at the output time (s)
  b      index of the frame (batch) the detection belongs to
Properties per detection (11 values with the default harmonics):
  r / 60 | cos psi, sin psi | Doppler as a CIRCULAR value: sin / cos of k pi vr / v_max for k in harmonics (a radar measures
  Doppler only modulo its unambiguous interval, so +v_max and -v_max are neighbours) | rcs / 20 | dt / 0.5
Raster channels per cell of the grid (x forward, y left, 0.5 m, same grid and index order as the camera BEV (x, y)):
  count | mean of the properties of the detections in the cell | mean sub-cell offset (dx, dy in cell units): nothing is lost
  by binning.  Cells without a detection are zero.

Tests: tests/test_radar_bev.py
"""
import math

import torch
import torch.nn as nn

try:
    from net.model.lift_pd import Grid
except ImportError:  # imported as a package module
    from .lift_pd import Grid

V_MAX = 30.0                 # unambiguous Doppler interval half-width in m/s (sensor property; nuScenes: 30 above the observed range)
HARMONICS = (1, 2, 4)        # all integers: the code stays periodic with the interval
RANGE_SCALE, RCS_SCALE, AGE_SCALE = 60.0, 20.0, 0.5


def to_vehicle(r, phi, sx, sy, yaw):
    """Sensor polar (range, azimuth in the sensor frame) -> vehicle-frame position and ray direction psi."""
    psi = phi + yaw
    return sx + r * torch.cos(psi), sy + r * torch.sin(psi), psi


def doppler_code(vr, v_max=V_MAX, harmonics=HARMONICS):
    """(n,) m/s -> (n, 2 * len(harmonics)): sin / cos of k pi vr / v_max. vr and vr + 2 v_max have the same code."""
    a = math.pi * vr / v_max
    return torch.cat([torch.stack([torch.sin(k * a), torch.cos(k * a)], -1) for k in harmonics], -1)


def properties(det, v_max=V_MAX, harmonics=HARMONICS):
    return torch.cat([(det["r"] / RANGE_SCALE)[:, None], torch.cos(det["psi"])[:, None], torch.sin(det["psi"])[:, None],
                      doppler_code(det["vr"], v_max, harmonics), (det["rcs"] / RCS_SCALE)[:, None],
                      (det["dt"] / AGE_SCALE)[:, None]], -1)


def n_properties(harmonics=HARMONICS):
    return 5 + 2 * len(harmonics)


def rasterize(det, B, grid=None, v_max=V_MAX, harmonics=HARMONICS):
    """Detections -> (B, 1 + P + 2, NX, NY): count, mean properties, mean sub-cell offset. No learned parameters."""
    grid = grid or Grid()
    NX, NY, res = grid.nx, grid.ny, grid.res
    P = n_properties(harmonics)
    fx, fy = (det["x"] - grid.x_min) / res, (det["y"] - grid.y_min) / res
    ix, iy = fx.floor().long(), fy.floor().long()
    inside = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
    vals = torch.cat([torch.ones_like(fx)[:, None], properties(det, v_max, harmonics),
                      (fx - ix - 0.5)[:, None], (fy - iy - 0.5)[:, None]], -1)[inside]
    flat = (det["b"][inside] * NX + ix[inside]) * NY + iy[inside]
    acc = torch.zeros(B * NX * NY, 1 + P + 2, device=vals.device, dtype=vals.dtype)
    acc.index_add_(0, flat, vals)
    cnt = acc[:, :1]
    acc = torch.cat([cnt, acc[:, 1:] / cnt.clamp(min=1)], 1)
    return acc.view(B, NX, NY, 1 + P + 2).permute(0, 3, 1, 2).contiguous()


class RadarBEV(nn.Module):
    """Raster + two 3x3 convolutions -> (B, out_channels, NX, NY), the grid of the camera BEV."""

    def __init__(self, grid=None, out_channels=32, v_max=V_MAX, harmonics=HARMONICS):
        super().__init__()
        self.grid, self.v_max, self.harmonics = grid or Grid(), v_max, harmonics
        cin = 1 + n_properties(harmonics) + 2
        self.net = nn.Sequential(nn.Conv2d(cin, out_channels, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU())

    def forward(self, det, B):
        return self.net(rasterize(det, B, self.grid, self.v_max, self.harmonics))


# ----------------------------------------------------------------------------------------------- self-test
