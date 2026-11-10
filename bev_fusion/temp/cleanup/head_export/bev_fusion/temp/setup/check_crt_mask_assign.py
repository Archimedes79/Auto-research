"""One-off: does `grid[mask][:, 0:2] = value` (CRT-Fusion crtfusion.py about line 413) change `grid`?

Boolean-mask indexing in PyTorch returns a copy; an assignment into a slice of that copy is lost.
Run: wsl ~/miniconda3/envs/gcar/bin/python temp/setup/check_crt_mask_assign.py
"""
import torch

grid = torch.zeros(5, 4)
mask = torch.tensor([True, False, True, False, False])
grid[mask][:, 0:2] = 7.0                      # the pattern in the released code
print("after grid[mask][:, 0:2] = 7:", "unchanged" if grid.abs().sum() == 0 else "changed")
grid[mask, 0:2] = 7.0                         # the in-place form that would have worked
print("after grid[mask, 0:2] = 7:", "changed" if grid.abs().sum() > 0 else "unchanged")
