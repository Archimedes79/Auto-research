"""E377 layer 1 (scratch, CPU): the section near_none of `jtoken_eval.py anatomy` on e373_cb_s0 against the stored scratch jsons temp/e377/out/near_e373_cb_s0_<split>_every<N>.json (R 2.0) and dup_..json, and its two new
numbers (deg3 touching a far none token's link, the median p_none of near / far) against the scratch way of e377_near.py (norm, matmul, sets) on the same dump. The anatomy json goes to temp/e377/out_e377a/.
usage: bash tools/env/_py14.sh e377a_layer1.py <split> <every> [<folder of jtoken_eval.py>]"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
split, every = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
if len(sys.argv) > 3:
    sys.path.insert(0, os.path.abspath(sys.argv[3]))
import jtoken_eval as E  # noqa: E402

print("jtoken_eval:", E.__file__, flush=True)
out = os.path.join(ROOT, "temp", "e377", "out_e377a")
os.makedirs(out, exist_ok=True)
a = E.parse_args(["anatomy", "--run", "e373_cb_s0", "--seed", "0", "--split", split, "--every", str(every)])
path = E.anatomy(a, out_dir=out)
nn = json.load(open(path))["near_none"]
c, pf, sh = nn["counts"], nn["per_frame"], nn["shares"]
near = json.load(open(os.path.join(ROOT, "temp", "e377", "out", f"near_e373_cb_s0_{split}_every{every}.json")))
dup = json.load(open(os.path.join(ROOT, "temp", "e377", "out", f"dup_e373_cb_s0_{split}_every{every}.json")))
R, T, D = near["per_R"]["2.0"], near["totals"], dup["counts"]
pairs = dict(frames=(nn["frames"], T["frames"]), none=(c["none"], T["none"]), dup=(c["dup"], T["dup"]), kept_dup=(c["kept_dup"], T["dup_kept"]), deg3=(c["deg3"], T["deg3"]),
             near_per_frame=(pf["near"], R["near_per_frame"]), far_per_frame=(pf["far"], R["far_per_frame"]), kept_near_share=(sh["kept_near"], R["near_kept_share"]), kept_far_share=(sh["kept_far"], R["far_kept_share"]),
             dup_near_share=(sh["dup_near"], R["dup_covered"]), near_that_are_dup=(c["dup_near"] / max(c["near"], 1), R["near_that_are_dup"]), deg3_touch_near=(sh["deg3_touch_near"], R["deg3_touch_near"]),
             dup_per_frame=(pf["dup"], near["dup_per_frame"]), kept_dup_share=(sh["kept_dup"], near["dup_kept_share"]), none_per_frame=(pf["none"], near["none_per_frame"]), deg3_per_frame=(pf["deg3"], near["deg3_per_frame"]),
             dup_frames=(nn["frames"], D["frames"]), dup_none_label=(c["none"], D["none_label"]), dup_dup=(c["dup"], D["dup"]), dup_kept_dup=(c["kept_dup"], D["kept_dup"]), dup_deg3=(c["deg3"], D["deg3"]),
             dup_deg3_touch_dup=(c["deg3_touch_dup"], D["deg3_touch_dup"]), dup_share_deg3_touch_dup=(sh["deg3_touch_dup"], dup["shares"]["deg3_touch_dup"]), dup_share_dup_kept=(sh["kept_dup"], dup["shares"]["dup_kept"]))
# the two new numbers the stored jsons do not hold: the scratch way of e377_near.py on the same dump
r = E.read_dump(argparse.Namespace(run="e373_cb_s0", model="model.pt", seed=0, split=split, every=every, drop=0.0, drop_seed=0), E.RUNS, 0)
far_touch, deg3, pn_near, pn_far = 0, 0, [], []
for i, f, t, v in zip(r["idx"], r["fr"], r["toks"], r["valid"]):
    it = r["ds"][int(i)]
    yt, v = np.asarray(it["ytype"]), np.asarray(v, bool)
    pos, dirv = np.asarray(t["pos"], float), np.asarray(t["dirv"], float)
    none, asg = v & (yt == 0), v & (yt != 0)
    nr = none & ((np.linalg.norm(pos[:, None] - pos[None], axis=2) <= 2.0) & (dirv @ dirv.T >= 0.5) & asg[None, :]).any(1)
    fa = none & ~nr
    cc = E.decode_variant(f, v, 0.3, 0.6, False, 0.2)["rec"]
    s, d = cc["src"].astype(int), cc["dst"].astype(int)
    jp = np.flatnonzero(np.bincount(s, minlength=len(v)) + np.bincount(d, minlength=len(v)) >= 3)
    tf = fa[s] | fa[d]
    nd = set(s[tf].tolist()) | set(d[tf].tolist())
    far_touch += sum(int(x) in nd for x in jp)
    deg3 += len(jp)
    pn = np.asarray(f["pnone"], float)
    pn_near.append(pn[nr])
    pn_far.append(pn[fa])
pairs.update(scratch_deg3=(c["deg3"], deg3), scratch_deg3_touch_far=(c["deg3_touch_far"], far_touch), scratch_median_pnone_near=(nn["median_pnone"]["near"], float(np.median(np.concatenate(pn_near)))),
             scratch_median_pnone_far=(nn["median_pnone"]["far"], float(np.median(np.concatenate(pn_far)))))
bad = {k: x for k, x in pairs.items() if x[0] != x[1]}
for k, (g, w) in pairs.items():
    print(f"  {k:<28} {g!r:<24} {w!r:<24} {'equal' if g == w else 'DIFF ' + str(g - w)}")
print(f"layer 1 {split} every {every}: {len(pairs) - len(bad)} of {len(pairs)} numbers equal" + ("" if not bad else f", DIFFERENT: {sorted(bad)}"))
print("anatomy json:", os.path.relpath(path, ROOT))
sys.exit(1 if bad else 0)
