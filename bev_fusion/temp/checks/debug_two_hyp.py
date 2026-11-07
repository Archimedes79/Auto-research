import sys, torch
sys.path.insert(0, "training"); sys.path.insert(0, "tests")
import test_temporal as T
from temporal import TemporalPropagation
tm = TemporalPropagation(cin=1, grid=T.G, n_hyp=1, static_hyp=True).double()
tm.set_velocity((4.0, 0.0), static_weight=0.0)
r = tm(T.spike(), [(0.5, 0.0, 0.0)])
f = r["feat"][0, 0]; m = r["mass"][0, 0]
print("pi at src", r["pi"][0, :, 20, 20].tolist(), "v_hyp", r["v_hyp"][0, :, :, 20, 20].tolist())
nz = (f.abs() > 1e-6).nonzero().tolist()
print((f[20,20]-0.5).item(), (f[24,20]-0.5).item()); print([(i, j, round(f[i, j].item(), 4), round(m[i, j].item(), 4)) for i, j in nz])
