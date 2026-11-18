import os, sys, time, numpy as np
R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", "evaluation", ".."):
    sys.path.insert(0, os.path.join(R, sub))
import lane_map_data as LMD
ds = LMD.MapData("train", 0, "e292_a", None, "train", init_seed=0)
idx = np.linspace(0, len(ds) - 1, 40).astype(int)
t0 = time.time(); n = 0; sizes = []
for i in idx:
    for _ in range(2):
        r = ds.raw_map(int(i)); sizes.append(len(r[0])); n += 1
dt = (time.time() - t0) / n
print(f"train-mode map per read: {dt*1000:.0f} ms (1 process, loaded box); x8 frames per step / 4 workers = {dt*8/4*1000:.0f} ms per step; mean samples {np.mean(sizes):.0f}")
# scene structure: how many scenes, are consecutive scenes from the same place
print("scenes", len(set(ds.scene.tolist())), "frames", len(ds.scene))
