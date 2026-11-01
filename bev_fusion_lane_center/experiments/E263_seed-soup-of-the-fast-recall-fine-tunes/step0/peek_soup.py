"""read-only: buffers / BN keys of the j1rc checkpoints and the relative weight distance seed-to-seed and seed-to-start"""
import sys, torch
torch.set_num_threads(2)
root = sys.argv[1]
runs = ["e123_D7_r3_j1", "e18_D7_j1rc", "e18_D7_j1rcs1", "e24_D7_j1rcs2", "e24_D7_j1rcs3", "e123_D7_r3_f1", "e123_D7_r3_rc1", "e18_D7_rc1s1"]
sd = {r: torch.load(f"{root}/runs/e123/{r}/model.pt", map_location="cpu") for r in runs}
k0 = sd[runs[0]]
print("keys", len(k0), "running:", [k for k in k0 if "running" in k][:6], "num_batches:", sum("num_batches" in k for k in k0))
def flat(s):
    return torch.cat([v.float().flatten() for k, v in s.items() if v.dtype.is_floating_point and "running" not in k])
F = {r: flat(sd[r]) for r in runs}
def rel(a, b):
    return float((F[a] - F[b]).norm() / F[b].norm())
print("j1 start norm", float(F["e123_D7_r3_j1"].norm()))
for r in runs[1:5]:
    print("from j1 start", r, round(rel(r, "e123_D7_r3_j1"), 4))
for i in range(1, 5):
    for j in range(i + 1, 5):
        print("seed-seed", runs[i], runs[j], round(rel(runs[i], runs[j]), 4))
print("f1 start -> rc1", round(rel("e123_D7_r3_rc1", "e123_D7_r3_f1"), 4), "rc1s1", round(rel("e18_D7_rc1s1", "e123_D7_r3_f1"), 4), "rc1-rc1s1", round(rel("e123_D7_r3_rc1", "e18_D7_rc1s1"), 4))
print("j1 vs f1", round(rel("e123_D7_r3_j1", "e123_D7_r3_f1"), 4))
