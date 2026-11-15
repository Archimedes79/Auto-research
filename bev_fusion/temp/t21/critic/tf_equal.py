"""critic: the new TemporalFusion.forward == the old one (git HEAD) bit for bit for every step form the window path passes"""
import sys
import torch
torch.set_num_threads(1)
ROOT = "/mnt/c/Auto-research/bev_fusion"
sys.path.insert(0, ROOT + "/training")
sys.path.insert(0, ROOT + "/temp/t21/critic")
from lift_pd import Grid
import temporal_fusion as new
import old_temporal_fusion as old

grid = Grid(x_min=0.0, x_max=16.0, y_min=-8.0, y_max=8.0, res=0.5)
B, C = 2, 8
X, Y = grid.nx, grid.ny
torch.manual_seed(0)
mn = new.TemporalFusion(C, C, grid, mem_cells=(64, 64)).train()
mo = old.TemporalFusion(C, C, grid, mem_cells=(64, 64)).train()
mo.load_state_dict(mn.state_dict())
cur = torch.randn(B, C, X, Y)
past = [(torch.randn(B, C, X, Y), [(torch.full((B,), 1 / 12), torch.tensor([8.0, 9.0]), torch.tensor([0.1, -0.2]))] * 2)]
obs = (torch.rand(B, 1, X, Y) > 0.3).float()
forms = {
    "None": None,
    "floats": (0.5, 9.0, 0.1),
    "tensor(B)": (torch.tensor([0.5, 0.45]), torch.tensor([8.0, 10.0]), torch.tensor([0.05, -0.1])),
    "tensor(1)": (torch.tensor([0.5]), torch.tensor([8.0]), torch.tensor([0.05])),
    "list3": [0.5, 9.0, 0.1],
    "tensor3": torch.tensor([0.5, 9.0, 0.1]),
    "0d": (torch.tensor(0.5), torch.tensor(9.0), torch.tensor(0.1)),
}
st0 = None
bad = []
for name, step in forms.items():
    o1, s1, _ = mo(cur, past, None, None, obs)          # warm state from a cold call
    st_old = {k: v.clone() for k, v in s1.items()}
    st_new = {k: v.clone() for k, v in s1.items()}
    try:
        a = mo(cur, past, st_old, step, obs)
    except Exception as e:
        a = e
    try:
        b = mn(cur, past, st_new, step, obs)
    except Exception as e:
        b = e
    if isinstance(a, Exception) or isinstance(b, Exception):
        print(name, "old:", type(a).__name__ if isinstance(a, Exception) else "ok", "new:", type(b).__name__ if isinstance(b, Exception) else "ok", (a if isinstance(a, Exception) else b))
        continue
    same = torch.equal(a[0], b[0]) and all(torch.equal(a[1][k], b[1][k]) for k in a[1])
    print(f"{name:10s} identical: {same}")
    if not same:
        bad.append(name)
print("DIFFERENT:", bad)
