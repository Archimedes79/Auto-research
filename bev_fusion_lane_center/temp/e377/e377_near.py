"""E377 v2 card measurement (scratch, CPU): the none tokens NEAR an assigned token (within R m and cos >= 0.5 of a valid token with a type label), for R in 1.5 / 2 / 3 m,
their overlap with the labelled duplicates (none tokens in some G(i)), their kept share at the D115 cell and the token degree >= 3 points they touch.
usage: bash tools/env/_py14.sh e377_near.py <run> <seed> <split> <every>"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jtoken_eval as E  # noqa: E402

run, seed, split, every = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
a = argparse.Namespace(run=run, model="model.pt", seed=seed, split=split, every=every, drop=0.0, drop_seed=0)
r = E.read_dump(a, E.RUNS, 0)
ds, idx = r["ds"], r["idx"]
RS = (1.5, 2.0, 3.0)
acc = {R: dict(near=0, near_dup=0, near_kept=0, far_kept=0, far=0, deg3_touch_near=0) for R in RS}
tot = dict(frames=0, none=0, dup=0, dup_kept=0, deg3=0)
for k, (i, f, t, v) in enumerate(zip(idx, r["fr"], r["toks"], r["valid"])):
    it = ds[int(i)]
    y = np.asarray(it["y"]) > 0
    yt = np.asarray(it["ytype"])
    v = np.asarray(v, bool)
    pos, dirv = np.asarray(t["pos"], float), np.asarray(t["dirv"], float)
    none = v & (yt == 0)
    asg = v & (yt != 0)
    dup = none & y.any(0)
    dist = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    cos = dirv @ dirv.T
    fr = E.decode_variant(f, v, 0.3, 0.6, False, 0.2)
    keep, c = fr["keep"], fr["rec"]
    s, d = c["src"].astype(int), c["dst"].astype(int)
    deg = np.bincount(s, minlength=len(keep)) + np.bincount(d, minlength=len(keep))
    jp = np.flatnonzero(deg >= 3)
    tot["frames"] += 1
    tot["none"] += int(none.sum())
    tot["dup"] += int(dup.sum())
    tot["dup_kept"] += int((dup & keep).sum())
    tot["deg3"] += len(jp)
    for R in RS:
        near = none & ((dist <= R) & (cos >= 0.5) & asg[None, :]).any(1)
        far = none & ~near
        acc[R]["near"] += int(near.sum())
        acc[R]["near_dup"] += int((near & dup).sum())
        acc[R]["near_kept"] += int((near & keep).sum())
        acc[R]["far"] += int(far.sum())
        acc[R]["far_kept"] += int((far & keep).sum())
        tn = near[s] | near[d]
        nd = set(s[tn].tolist()) | set(d[tn].tolist())
        acc[R]["deg3_touch_near"] += sum(int(x) in nd for x in jp)
F = max(tot["frames"], 1)
out = dict(run=run, seed=seed, split=split, every=every, totals=tot,
           per_R={str(R): dict(near_per_frame=x["near"] / F, near_kept_share=x["near_kept"] / max(x["near"], 1), dup_covered=x["near_dup"] / max(tot["dup"], 1), near_that_are_dup=x["near_dup"] / max(x["near"], 1),
                               far_per_frame=x["far"] / F, far_kept_share=x["far_kept"] / max(x["far"], 1), deg3_touch_near=x["deg3_touch_near"] / max(tot["deg3"], 1),
                               mean_none_weight_if_near_1=(x["near"] + 0.25 * x["far"]) / max(tot["none"], 1),
                               w_near_for_mean_0p5=(0.5 * tot["none"] - 0.25 * x["far"]) / max(x["near"], 1)) for R, x in acc.items()},
           dup_per_frame=tot["dup"] / F, dup_kept_share=tot["dup_kept"] / max(tot["dup"], 1), none_per_frame=tot["none"] / F, deg3_per_frame=tot["deg3"] / F)
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(ROOT, "temp", "e377", "out", f"near_{run}_{split}_every{every}.json"), "w"), indent=1)
