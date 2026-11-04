"""Read-only: cell-collision statistics of nuScenes-mini radar detections in the 0.5 m BEV grid (x 0..60, y -26..26).
Input: temp/radar/cache/mini_multisweep.pt (made by temp/radar/radar_multisweep_cache.py). Prints a table; writes nothing."""
import numpy as np
import torch

frames = torch.load("temp/radar/cache/mini_multisweep.pt")
print("frames", len(frames))


def stats(sel_fn, res, label):
    tot = 0
    in_multi = 0
    cells = 0
    cells_ge = {2: 0, 3: 0, 4: 0}
    lost = {1: 0, 2: 0, 3: 0, 4: 0, 8: 0}
    mx = 0
    per_frame_pts = []
    for f in frames:
        sel = sel_fn(f)
        x, y = f["x"][sel].numpy(), f["y"][sel].numpy()
        ins = (x >= 0) & (x < 60) & (y >= -26) & (y < 26)
        ix = np.floor(x[ins] / res).astype(np.int64)
        iy = np.floor((y[ins] + 26) / res).astype(np.int64)
        key = ix * 10000 + iy
        u, c = np.unique(key, return_counts=True)
        tot += len(key)
        per_frame_pts.append(len(key))
        cells += len(u)
        in_multi += c[c >= 2].sum()
        for k in cells_ge:
            cells_ge[k] += int((c >= k).sum())
        for k in lost:
            lost[k] += int(np.clip(c - k, 0, None).sum())
        mx = max(mx, int(c.max()) if len(c) else 0)
    print(f"{label} @ {res} m: detections in grid per frame (median) {int(np.median(per_frame_pts))}, total {tot}; occupied cells {cells}; "
          f"detections sharing a cell {100 * in_multi / tot:.1f} %; cells with >=2/3/4 detections: "
          + "/".join(f"{100 * cells_ge[k] / cells:.1f}%" for k in (2, 3, 4)) + f"; max in a cell {mx}; "
          + "dropped by K-slot cap K=1/2/3/4/8: " + "/".join(f"{100 * lost[k] / tot:.1f}%" for k in (1, 2, 3, 4, 8)))


one_sweep = lambda f: (f["dt"] > -0.0385) & (f["dt"] <= 0.0385)
all_sweeps = lambda f: torch.ones_like(f["dt"], dtype=torch.bool)
for res in (0.5, 0.25):
    stats(one_sweep, res, "1 sweep per radar (|dt|<0.0385)")
    stats(all_sweeps, res, "all cached sweeps (-0.1..0.5 s)")
# how many sweeps does the window hold per radar: detections per frame
n = [len(f["x"]) for f in frames]
print("detections per frame, all sweeps: median", int(np.median(n)), "max", max(n))
