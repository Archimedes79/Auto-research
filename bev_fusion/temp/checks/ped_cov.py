import random, sys
sys.path.insert(0, "tests"); sys.path.insert(0, "training")
from lift_pd import Grid
from synthetic_world import gt_bev, random_scene
G = Grid()
n_ped, c50, c25, c05 = 0, 0, 0, 0
for i in range(40):
    sc = random_scene(random.Random(i))
    peds = [b for b in sc.boxes if b.cls == "pedestrian" and 0 < b.cx < 60 and abs(b.cy) < 26]
    if not peds:
        continue
    g = gt_bev(random_scene(random.Random(i)), G)["pedestrians"]
    n_ped += len(peds)
    c50 += int((g > 0.5).sum()); c25 += int((g > 0.25).sum()); c05 += int((g > 0.05).sum())
print(f"{n_ped} pedestrians: cells > 0.5: {c50}, > 0.25: {c25}, > 0.05: {c05}")
