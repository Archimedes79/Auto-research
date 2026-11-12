"""E332 / E333 kill check: exit 1 when the maps of a simplification arm have a keypoint F1 (thr 0.3, 3 m, distance only, mean of entry and exit) more than <tol> below E307 of the same seed.
usage: kill_check.py <maps run name e.g. e332_ma_s0> <seed> [tol 0.05]   (run from any folder: paths from BFLC_ROOT)"""
import json
import os
import sys

ROOT = os.environ["BFLC_ROOT"]
run, S = sys.argv[1], sys.argv[2]
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05


def f(r):
    k = json.load(open(os.path.join(ROOT, "runs", r, "jmaps_val.json")))["keypoints"]
    return sum(k[f"kp_ours_tol3.0_noang_{t}"]["f1"] for t in ("ent", "ext")) / 2


a, b = f(run), f(f"e307_n_s{S}")
print(f"seed {S}: kp F1 of {run} {a:.4f}, E307 {b:.4f}, difference {a - b:+.4f} (kill below -{tol})")
sys.exit(1 if a < b - tol else 0)
