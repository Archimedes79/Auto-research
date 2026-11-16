"""Scratch: a toy run written by trainer.fit, read by `python -m trainsuite status / health` (the contract end to end)."""
import os
import sys

import torch

import trainer as T

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cc_runs")
os.makedirs(root, exist_ok=True)
torch.manual_seed(0)
m = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
W = torch.randn(8, 1)


def batch_fn(step):
    g = torch.Generator().manual_seed(step)
    x = torch.randn(64, 8, generator=g)
    return x, x @ W


def loss_fn(model, b):
    l = torch.nn.functional.mse_loss(model(b[0]), b[1])
    return l, {"group.mse": float(l)}


T.fit(m, loss_fn, T.Run("toy", root), steps=200, batch_fn=batch_fn, lr=3e-3, amp="none", device="cpu", ckpt_every=50, log_every=20, probe=True)
print("root", root)
