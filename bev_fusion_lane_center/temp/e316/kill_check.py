"""E316 kill check: prints the keypoint F1 (thr 0.3, 3 m, distance only, mean of entry and exit) of the maps after the joint pair training against E307 of the same seed; exit 1 when more than 0.05 lower."""
import json, sys
bad = 0
for S in (0, 1):
    f = lambda r: sum(json.load(open(f"runs/{r}/jmaps_val.json"))["keypoints"][f"kp_ours_tol3.0_noang_{k}"]["f1"] for k in ("ent", "ext")) / 2
    a, b = f(f"e316_p_s{S}_maps"), f(f"e307_n_s{S}")
    print(f"seed {S}: kp F1 maps after the joint pair training (6000 steps) {a:.4f}, E307 {b:.4f}, difference {a - b:+.4f}")
    bad |= a < b - 0.05
sys.exit(int(bad))
