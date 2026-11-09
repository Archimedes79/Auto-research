"""Layer-0 check of the footprint-share measurement (known answers on a synthetic mask). Run from bev_fusion (WSL, gcar python)."""
import sys

import numpy as np

sys.path.insert(0, "temp/coverage")
from footprint_share import NX, NY, RES, X0, Y0, share  # noqa: E402

mask = np.zeros((NX, NY), np.float32)
mask[:, NY // 2:] = 1.0                                         # the left half of the grid (y >= 0) is observed
box = lambda cx, cy, l, w: np.array([[cx - l / 2, cy - w / 2], [cx + l / 2, cy - w / 2], [cx + l / 2, cy + w / 2], [cx - l / 2, cy + w / 2]])
checks = {
    "fully inside": (box(20.0, 10.0, 4.0, 2.0), 1.0),
    "fully outside": (box(20.0, -10.0, 4.0, 2.0), 0.0),
    "half over the border y = 0": (box(20.0, 0.0, 4.0, 2.0), 0.5),
    "a quarter inside (4 x 2 m, border at 1.5 m from the lower side)": (box(20.0, 1.0 - 1.5 + 0.0, 4.0, 2.0), None),
}
ok = True
for name, (poly, want) in checks.items():
    f, ing = share(poly, mask)
    if want is None:
        continue
    good = abs(f - want) < 0.02 and ing == 1.0
    ok &= good
    print(f"{'ok ' if good else 'BAD'} {name}: f = {f:.3f} (want {want}), inside the grid {ing:.2f}")
# a box cut by the far edge of the grid (x_max = 60): the half outside counts as not observed
f, ing = share(box(60.0, 10.0, 4.0, 2.0), mask)
good = abs(f - 0.5) < 0.02 and abs(ing - 0.5) < 0.02
ok &= good
print(f"{'ok ' if good else 'BAD'} cut by the far edge: f = {f:.3f} (want 0.5), inside the grid {ing:.2f}")
# a rotated box (45 degrees) fully inside
c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
r = np.array([[c, -s], [s, c]])
poly = (box(0, 0, 4.0, 2.0) @ r.T) + np.array([30.0, 12.0])
f, ing = share(poly, mask)
good = abs(f - 1.0) < 0.02
ok &= good
print(f"{'ok ' if good else 'BAD'} rotated box inside: f = {f:.3f}")
# a 0.6 x 0.9 m pedestrian sliding over the mask border: the share follows the true area fraction; the sample grid (0.125 m, 5 samples across 0.6 m) limits it to a granularity of 0.2
errs = []
for yc in np.arange(-0.4, 0.4001, 0.01):
    f, _ = share(box(20.0, yc, 0.9, 0.6), mask)
    true = float(np.clip((yc + 0.3) / 0.6, 0.0, 1.0))
    errs.append(abs(f - true))
good = max(errs) <= 0.2 + 1e-6 and float(np.mean(errs)) < 0.06
ok &= good
print(f"{'ok ' if good else 'BAD'} pedestrian sliding over the border: max |f - true| = {max(errs):.3f}, mean {float(np.mean(errs)):.3f} (granularity 0.2)")
# a car (4.5 x 1.9 m) sliding over the border: finer
errs = []
for yc in np.arange(-1.2, 1.2001, 0.01):
    f, _ = share(box(20.0, yc, 4.5, 1.9), mask)
    true = float(np.clip((yc + 0.95) / 1.9, 0.0, 1.0))
    errs.append(abs(f - true))
good = max(errs) <= 0.07 + 1e-6
ok &= good
print(f"{'ok ' if good else 'BAD'} car sliding over the border: max |f - true| = {max(errs):.3f}, mean {float(np.mean(errs)):.3f}")
print("selfcheck", "passed" if ok else "FAILED")
sys.exit(0 if ok else 1)
