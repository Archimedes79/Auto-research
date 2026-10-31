"""scratch: check the factual claims of the captions against the stored GT."""
import sys

import numpy as np

sys.path.insert(0, ".")
import autogt  # noqa: E402

for name, k in (("scene-0128", 34), ("scene-0094", 0), ("scene-0131", 8), ("scene-0361", 4), ("scene-0479", 2), ("scene-0317", 2), ("scene-0194", 14), ("scene-0286", 38)):
    tok = autogt.scene_tokens(name, "v1.0-trainval")[k]
    g = autogt.load(tok)
    print(name, k)
    for b in g["meta"]["boxes"]:
        sp = float(np.hypot(*b["vel"]))
        if b["observed"] and (sp > 5 or b["vel_flag"] in ("radar_conflict", "radar_override")):
            cx, cy, cz = b["center"]
            print(f"   {b['group']:11s} x {cx:6.1f} y {cy:6.1f} zbottom {cz - b['size'][2] / 2:5.1f}  {sp * 3.6:5.1f} km/h flag {b['vel_flag']} src {b['vel_source']} shift {np.hypot(*b['shift']):.2f}")
