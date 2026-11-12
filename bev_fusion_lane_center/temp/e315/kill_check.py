"""E315 kill check: exit 1 when the maps after 6000 steps have a keypoint F1 (thr 0.3, 3 m, distance only, mean of entry and exit) more than 0.03 below E307 of the same seed.  usage: kill_check.py <seed>"""
import json, sys
S = sys.argv[1]
f = lambda r: sum(json.load(open(f"../../runs/{r}/jmaps_val.json"))["keypoints"][f"kp_ours_tol3.0_noang_{k}"]["f1"] for k in ("ent", "ext")) / 2
a, b = f(f"e315_m_s{S}"), f(f"e307_n_s{S}")
print(f"seed {S}: kp F1 maps 6000 steps {a:.4f}, E307 {b:.4f}, difference {a - b:+.4f}")
sys.exit(1 if a < b - 0.03 else 0)
