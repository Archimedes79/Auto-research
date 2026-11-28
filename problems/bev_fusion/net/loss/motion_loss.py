"""The loss of the explicit velocity head."""
import torch


def motion_loss(v_hat, v_gt, object_mask, min_speed=1.0, w_background=0.1, weight=None):
    """L1 to the GT velocity on object cells (GT set to 0 below min_speed: a parked car does not move) plus a weak L1 to 0
    on the other cells (the background is static). v_hat, v_gt (B, 2, X, Y); object_mask (B, 1, X, Y) bool / {0, 1}.
    weight (B, 1, X, Y) or None: a per-cell weight (LossConfig.hygiene_2026_11_02: w_obj with w_interpolated, w_unknown); each term is then a weighted mean;
    None = the plain means, bit for bit."""
    m = object_mask.to(v_hat.dtype)
    v_gt = torch.where(v_gt.norm(dim=1, keepdim=True) < min_speed, torch.zeros_like(v_gt), v_gt)
    if weight is not None:
        w = weight.to(v_hat.dtype)
        obj = ((v_hat - v_gt).abs() * m * w).sum() / (m * w * 2).sum().clamp(min=1)
        bg = (v_hat.abs() * (1 - m) * w).sum() / ((1 - m) * w * 2).sum().clamp(min=1)
        return obj + w_background * bg
    obj = ((v_hat - v_gt).abs() * m).sum() / (m.sum() * 2).clamp(min=1)
    bg = (v_hat.abs() * (1 - m)).sum() / ((1 - m).sum() * 2).clamp(min=1)
    return obj + w_background * bg
