"""read-only: which tensors carry the seed-to-seed distance of the j1rc fine-tunes"""
import sys, torch
torch.set_num_threads(2)
root = sys.argv[1]
L = lambda r: torch.load(f"{root}/runs/e123/{r}/model.pt", map_location="cpu")
j1, a, b, f1, r1 = L("e123_D7_r3_j1"), L("e18_D7_j1rc"), L("e18_D7_j1rcs1"), L("e123_D7_r3_f1"), L("e123_D7_r3_rc1")
rows = []
for k in j1:
    if not j1[k].dtype.is_floating_point:
        continue
    d_ab = float((a[k].float() - b[k].float()).norm() ** 2)
    d_s = float((a[k].float() - j1[k].float()).norm() ** 2)
    rows.append((d_ab, d_s, k, tuple(j1[k].shape), float(j1[k].float().norm()), float((r1[k].float() - f1[k].float()).norm())))
tot = sum(r[0] for r in rows)
rows.sort(reverse=True)
for r in rows[:14]:
    print(f"{r[2]:45s} {str(r[3]):18s} share_ab {r[0]/tot:.3f} |start| {r[4]:.2f} |a-start| {r[1]**0.5:.3f} |a-b| {r[0]**0.5:.3f} |rc1-f1| {r[5]:.3f}")
