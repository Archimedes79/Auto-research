"""The side loss of the delta-slot radar encoder and its weight over the steps."""
import os as _os
import sys as _sys
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # net/ (a flat run snapshot holds netpath.py next to this file)
import netpath  # noqa: E402,F401
import torch
import torch.nn.functional as F

from radar_encoder import slot_width


def aux_weight(step, start=1.0, floor=0.05, half_life=3000):
    """Weight of the side loss at a training step: starts at `start`, halves every `half_life` steps down to `floor` (0 switches the
    side loss off at the end). The task losses are the goal; this only keeps the first phase informative."""
    return floor + (start - floor) * 0.5 ** (step / half_life)


def slot_aux_loss(encoder, features, state, decoder=None):
    """Side loss: count (L1 on log1p), valid (BCE) and the tolerance-weighted L1 of the stored fields at valid slots."""
    out = (decoder or encoder.decoder)(features)
    W = slot_width(encoder.harmonics)
    loss = F.l1_loss(out[:, 0], torch.log1p(state[:, 0]))
    w = encoder.w_slot.view(1, -1, 1, 1)
    for k in range(encoder.K + 1):
        vi = 1 + k * W
        valid = (state[:, vi] > 0.5).float()
        loss = loss + F.binary_cross_entropy_with_logits(out[:, vi], valid)
        m = valid[:, None]
        loss = loss + ((out[:, vi + 1: vi + W] - state[:, vi + 1: vi + W]).abs() * w * m).sum() / (m.sum() * (W - 1)).clamp(min=1)
    return loss
