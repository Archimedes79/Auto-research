"""E369 probe (throw-away): `jtoken_eval.frame_record` / `sys_record` against temp/qual_report_1115/pass1.pkl on its first frames (GT pieces, maneuvers; system B's maneuver counts from its stored lines)"""
import os
import pickle
import sys

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jtoken_eval as E  # noqa: E402

p1 = pickle.load(open(os.path.join(ROOT, "temp", "qual_report_1115", "pass1.pkl"), "rb"))
frames = E.QE.val_frames()
n, bad = int(sys.argv[1]), []
for k in range(0, 873, 873 // n):
    ref, f = p1["frames"][k], frames[k]
    rec, mans = E.frame_record(f)
    same = (rec["k"], rec["token"], rec["row"], rec["scene"], rec["n_man"]) == (ref["k"], ref["token"], ref["row"], ref["scene"], ref["n_man"]) and len(rec["gt"]) == len(ref["gt"])
    same &= all(np.array_equal(a[q], b[q]) for a, b in zip(rec["gt"], ref["gt"]) for q in ("pts", "tier", "vis")) and all(a["kinds"] == b["kinds"] for a, b in zip(rec["gt"], ref["gt"]))
    same &= all(np.array_equal(a, b) for a, b in zip(rec["man_pts"], ref["man_pts"]))
    for name in ref["sys"]:
        s = E.sys_record(dict(pts=[np.asarray(x, float) for x in ref["sys"][name]["lines"]]), mans, f)
        same &= all(s[q] == ref["sys"][name][q] for q in ("man_ok", "man_n", "man_real", "junc_hit", "junc_near"))
    f._lane = f._ref = None
    if not same:
        bad.append(k)
print(f"frames compared {len(range(0, 873, 873 // n))}, differing {bad}")
