"""The toy network: three 3 x 3 convolutions, one logit per pixel (is it the disc?)."""
import torch
from torch import nn


class DiscNet(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.first = nn.Conv2d(1, width, 3, padding=1)
        self.mid = nn.Conv2d(width, width, 3, padding=1)
        self.head = nn.Conv2d(width, 1, 3, padding=1)

    def features(self, x):
        return torch.relu(self.mid(torch.relu(self.first(x))))

    def forward(self, x):
        """(B, 1, S, S) pictures -> (B, 1, S, S) logits"""
        return self.head(self.features(x))
