import pickle, sys, math
sys.path.insert(0, ".")
from evaluation.metrics import in_front
d = pickle.load(open("runs/t22_B_s0/eval/scene_records_n1000.pkl", "rb"))["det"]
rows = []
for sc, per in d.items():
    fr = per.get(0.0, [])
    n = [sum(1 for g in gts if in_front(g)) for _, gts in fr]
    mov = [sum(1 for g in gts if in_front(g) and math.hypot(g["vx"], g["vy"]) > 0.5) for _, gts in fr]
    if n:
        rows.append((sum(n) / len(n), sum(mov) / len(mov), len(fr), sc))
rows.sort(reverse=True)
print("scene, frames in the 1000-frame val sample, mean boxes in the window per frame, mean moving boxes")
for m, mv, nf, sc in rows[:8]:
    print(f"{sc}  frames {nf:3d}  boxes {m:5.1f}  moving {mv:4.1f}")
