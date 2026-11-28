"""Camera-radar fusion at camera time: plug-in modules in front of the BEV trunk.

Inputs: the temporally fused camera planes (B, c_cam, X, Y) and the radar planes moved to the same time (B, c_rad, X, Y);
each branch carries its own confidence channels (mass, dt), so "sensor absent" is simply an all-zero input (mass 0).
Output: (B, c_out, X, Y) for BEVTrunk(cin=c_out).

- ConcatFusion (main, BEVFusion ConvFuser style): concat -> 3x3 conv -> BatchNorm -> ReLU.
- drop_modality: modality dropout in training (per sample, with probability p exactly one of the two modalities is
  zeroed, never both), applied when `training`.
Tests: tests/test_fusion.py
"""
import torch
import torch.nn as nn


def drop_modality(cam, rad, p):
    """Zero the camera planes of a fraction p/2 of the samples and the radar planes of another p/2 (never both)."""
    B = cam.shape[0]
    r = torch.rand(B, device=cam.device)
    keep_cam = (r >= 0.5 * p).to(cam.dtype).view(B, 1, 1, 1)
    keep_rad = ((r < 0.5 * p) | (r >= p)).to(cam.dtype).view(B, 1, 1, 1)
    return cam * keep_cam, rad * keep_rad


def _conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))


class ConcatFusion(nn.Module):
    def __init__(self, c_cam, c_rad, c_out, p_drop=0.2):
        super().__init__()
        self.conv = _conv_bn_relu(c_cam + c_rad, c_out)
        self.p_drop = p_drop

    def forward(self, cam, rad):
        if self.training and self.p_drop > 0:
            cam, rad = drop_modality(cam, rad, self.p_drop)
        return self.conv(torch.cat([cam, rad], 1))
