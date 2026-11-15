"""t5: where are the labelled divider cells in the image? Share per distance band and per image row band (VAL frames)."""
import json, os, numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
fr = json.load(open(os.path.join(OUT, "frames.json")))
nt, nv = len(fr["train"]), len(fr["val"])
m = np.load(os.path.join(OUT, "feats", "mask.npy"), mmap_mode="r")[nt:nt + nv].astype(np.float64)
b = np.load(os.path.join(OUT, "feats", "band.npy"), mmap_mode="r")[nt:nt + nv]
tot = m.sum()
for k, name in ((-1, "no ground band"), (0, "2-20 m"), (1, "20-40 m"), (2, "40-60 m")):
    print(f"{name:15s} {100 * m[b == k].sum() / tot:5.1f} % of labelled cells")
rows = m.sum((0, 2))
for r0 in range(0, 64, 8):
    print(f"rows {r0:2d}-{r0 + 7:2d} {100 * rows[r0:r0 + 8].sum() / tot:5.1f} %")
