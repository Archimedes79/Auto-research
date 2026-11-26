"""The toy loss: binary cross entropy per pixel, with a weight on the disc pixels."""
import torch
from torch import nn


class DiscLoss(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits, target):
        return nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
