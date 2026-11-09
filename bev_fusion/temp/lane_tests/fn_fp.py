"""scratch: precision / recall of the lane_divider and road_divider PRESENCE head from stored scene records (val, 1000 frames): is the IoU lost to misses (fn) or to false cells (fp)?
python temp/lane_tests/fn_fp.py runs/<run>/eval/scene_records_n1000.pkl ...   (cnt rows = tolerances, columns = tp_p, fp, tp_g, fn)"""
import pickle
import sys

import numpy as np

sys.path[:0] = ["evaluation", "training"]
try:
    from metrics import TOLERANCES
except Exception as e:                                      # noqa: BLE001
    TOLERANCES = None
    print("TOLERANCES not importable:", e)
print("tolerances", TOLERANCES)
for path in sys.argv[1:]:
    seg = pickle.load(open(path, "rb"))["seg"]
    print("==", path, len(seg), "scenes")
    for layer in ("lane_divider", "road_divider", "lane_divider:zc"):
        tot = None
        for sc, layers in seg.items():
            for key, cell in layers.get(layer, {}).items():
                if key[2:] == ("all", "all") and key[0] == 0.0:
                    c = np.asarray(cell["cnt"], dtype=np.float64)
                    tot = c if tot is None else tot + c
        if tot is None:
            continue
        for i, row in enumerate(tot):
            tp_p, fp, tp_g, fn = row
            tol = TOLERANCES[i] if TOLERANCES is not None and i < len(TOLERANCES) else i
            p, r = tp_p / max(tp_p + fp, 1), tp_g / max(tp_g + fn, 1)
            tp = (tp_p + tp_g) / 2
            print(f"{layer:16s} tol {tol}: precision {p:.3f} recall {r:.3f} iou {tp / max(tp + fp + fn, 1):.3f}   fp {fp:.0f} fn {fn:.0f} tp_p {tp_p:.0f} tp_g {tp_g:.0f}")
