"""critic: bytes of the tensors an autograd graph keeps (saved for backward) for ONE temporal-block call at the real size, B=1.
window call: K=2 past frames; recurrent call: K=1 past (the fused state). CPU, 1 thread."""
import os, sys
import torch
torch.set_num_threads(1)
ROOT = "/mnt/c/Auto-research/bev_fusion"
sys.path.insert(0, ROOT + "/training")
from lift_pd import Grid
from temporal_fusion import TemporalFusion

C, B = 80, 1
grid = Grid()
X, Y = grid.nx, grid.ny
print("grid", X, Y)
torch.manual_seed(0)
mod = TemporalFusion(C, C, grid).train()
step = (torch.full((B,), 0.5), torch.full((B,), 9.0), torch.full((B,), 0.1))
chain = [(torch.full((B,), 1 / 12), torch.full((B,), 9.0), torch.full((B,), 0.1))] * 2


def measure(K):
    cur = torch.randn(B, C, X, Y)
    pasts = [(torch.randn(B, C, X, Y, requires_grad=True), chain * (k + 1)) for k in range(K)]
    # warm state with one cold call (outside the measurement)
    out0, st, _ = mod(torch.randn(B, C, X, Y), [], None, None, torch.ones(B, 1, X, Y))
    inputs = {t.untyped_storage().data_ptr() for t in [cur] + [p for p, _ in pasts] + list(st.values())}
    seen = {}

    def pack(t):
        s = t.untyped_storage()
        seen[s.data_ptr()] = s.nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out, st2, _ = mod(cur.requires_grad_(True), pasts, st, step, torch.ones(B, 1, X, Y))
    extra = sum(v for k, v in seen.items() if k not in inputs)
    outs = sum(t.numel() * 4 for t in (out, st2["M"], st2["W"]))
    return extra / 1e6, outs / 1e6


for K in (2, 1, 0):
    a, o = measure(K)
    print(f"K={K}: saved-for-backward (excl. inputs) {a:.0f} MB at B=1 -> {3 * a:.0f} MB at B=3; outputs+state {o:.0f} MB")
