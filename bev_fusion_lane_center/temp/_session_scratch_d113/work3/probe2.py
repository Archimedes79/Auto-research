import os, sys
import numpy as np
R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import sd_lib as SD
ys = np.arange(-20, 30.001, 0.5)
a = np.stack([np.full_like(ys, -1.75), ys], 1)
b = np.stack([np.full_like(ys, 1.75), ys], 1)
o = np.stack([np.full_like(ys, 9.0), ys[::-1]], 1)
x = np.arange(-30, 30.001, 0.5)
c = np.stack([x, np.full_like(x, 12.0)], 1)
sd = SD.build_sd(dict(pts=[a, b, o, c]), [])
for poly, cnt in sd["roads"]:
    print(len(poly), poly[0], poly[-1], cnt[:3], set(cnt.tolist()))
rng = np.random.RandomState(3)
d = SD.degrade_sd(sd, rng, drop=0.1)
for poly, cnt in d["roads"]:
    print("deg", len(poly), poly[0], poly[-1], set(cnt.tolist()))
