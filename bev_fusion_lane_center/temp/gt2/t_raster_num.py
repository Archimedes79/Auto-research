import os, sys
import numpy as np
sys.path.insert(0, "../../training/grid")
import jgt as J
from shapely.geometry import Polygon, box as sbox
rng = np.random.RandomState(0)
be = bl = nr = 0
for _ in range(60):
    ang = np.sort(rng.uniform(0, 2 * np.pi, 6))
    poly = np.stack([rng.uniform(-20, 20) + rng.uniform(2, 9) * np.cos(ang), rng.uniform(-20, 20) + rng.uniform(2, 9) * np.sin(ang)], 1)
    sp = Polygon(poly)
    if not sp.is_valid or sp.area < 2:
        continue
    cells = ((poly + 50.0))[:, ::-1]
    os.environ.pop("BFLC_RASTER", None); ml = np.zeros((100, 100), np.uint8); J.fill_poly(ml, cells)
    os.environ["BFLC_RASTER"] = "exact"; mm = np.zeros((100, 100), np.uint8); J.fill_poly(mm, cells)
    ref = np.zeros((100, 100), bool)
    b = sp.bounds
    for i in range(int(b[0] + 48), int(b[2] + 53)):
        for j in range(int(b[1] + 48), int(b[3] + 53)):
            ref[i, j] = sp.intersection(sbox(i - 50, j - 50, i - 49, j - 49)).area > 0.5
    be += int((mm.astype(bool) != ref).sum()); bl += int((ml.astype(bool) != ref).sum()); nr += int(ref.sum())
print("cells (area > 0.5):", nr, "wrong cells exact:", be, "legacy:", bl)
