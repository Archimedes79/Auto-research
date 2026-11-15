"""Scratch (read-only): the maps numbers of E307 / E312 / E315 / E320 / E324 maps heads that the E319 gate and card cite (keypoint F1, far entry recall).  usage: python3 temp/e319/_maps_ref.py"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for run in ("e307_n_s{s}", "e312_p_s{s}_maps", "e315_m_s{s}", "e320_a_s{s}_maps", "e324_s_s{s}_maps", "e313_s_s{s}"):
    for s in (0, 1):
        p = os.path.join(ROOT, "runs", run.format(s=s), "jmaps_val.json")
        if not os.path.exists(p):
            print(run.format(s=s), "no jmaps_val.json")
            continue
        d = json.load(open(p))
        k = d["keypoints"]
        kp = (k["kp_ours_tol3.0_noang_ent"]["f1"] + k["kp_ours_tol3.0_noang_ext"]["f1"]) / 2
        far = d["range_city_split"]["range_ours_ent_gt35"]["recall"]
        crowd = d["crowded_split"]["ours_ent"]["crowded_recall"]
        print(f"{run.format(s=s):24s} kp F1 {kp:.4f}  far entry recall {far:.4f}  crowded {crowd:.4f}")
