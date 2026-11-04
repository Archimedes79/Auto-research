import sys, torch
sys.argv += ["--quick"]
sys.path.insert(0, "tests"); sys.path.insert(0, "training")
import test_flow_explains as T
data = T.dataset([50_000 + i for i in range(16)], "test")
maps = data[0][0]
lane = maps[0, 0]
print("lane coverage values:", sorted(set(round(x, 3) for x in lane[lane > 0].tolist()))[:8])
print("cells > 0.05:", int((lane > 0.05).sum()), " > 0.3:", int((lane > 0.3).sum()))
m = T.Model().eval()
with torch.no_grad():
    flows, _, _ = m(maps[None, 0], maps[None, 1], [data[0][3]], vgt=data[0][1][None])
v = flows[-1][0]
mk = (lane > 0.05) & (maps[0, 1:].max(0).values < 0.1)
print("lane cells in the mask", int(mk.sum()), " mean |v| there", float(v.norm(dim=0)[mk].mean()))
