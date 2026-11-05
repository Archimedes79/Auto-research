import random, sys, torch
sys.argv += ["--quick", "--data", "nuscenes", "--gate-doppler", "--sharp", "--pyr", "--doppler", "--rawwarp", "3.0", "--prior", "0.03", "--smooth", "0.05"]
sys.path.insert(0, "tests"); sys.path.insert(0, "training")
import test_flow_explains as T
T.DEV = "cpu"
items = T.nuscenes_items(True, 16, with_v1=True)[:4]      # cached by training/scratch.py as temp/checks/flow_explains_nusc_v1_test_16.pt (built on the first call)
# rotate_into_next per sample == one at a time
v = torch.randn(2, 2, T.G.nx, T.G.ny)
st = [items[0][3], items[1][3]]
both = T.rotate_into_next(v, st)
one = torch.cat([T.rotate_into_next(v[i:i + 1], [st[i]]) for i in range(2)])
print("per-sample rotation == one at a time:", torch.allclose(both, one), " TAG", T.TAG)
m = T.Model()
print("gate params", sum(p.numel() for p in m.gate.parameters()))
T.train(m, items, 2, random.Random(0), bs=2)
m.eval()
print({k: round(x, 2) for k, x in T.evaluate_recursive(m, items[:2]).items()})
print("gate weights (fresh, carried) mean", m.last_gate.mean((0, 2, 3)).tolist())
