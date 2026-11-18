"""E377 card measurement (scratch, CPU): the DUPLICATE tokens of a head's dump = valid tokens whose type label is none (ytype 0, assigned to no GT node) but which
are a GT successor of some source (in some G(i), any class: a token near a real GT node that another token was assigned to). Kept duplicates, their links and the
token-level degree >= 3 points they touch, at the D115 cell (short 0.3 / long 0.2 / keep 0.6 / none gate off).
usage: bash tools/env/_py14.sh e377_dup.py <run> <seed> <split> <every>"""
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
n = dict(frames=0, valid=0, none_label=0, dup=0, other_none=0, kept=0, kept_assigned=0, kept_dup=0, kept_other_none=0,
         links=0, links_touch_dup=0, links_touch_other_none=0, deg3=0, deg3_touch_dup=0, deg3_touch_other_none=0, deg3_is_dup=0)
pn_dup, pn_other = [], []
for k, (i, f, v) in enumerate(zip(idx, r["fr"], r["valid"])):
    it = ds[int(i)]
    y = np.asarray(it["y"]) > 0
    yt = np.asarray(it["ytype"])
    v = np.asarray(v, bool)
    none = v & (yt == 0)
    dup = none & y.any(0)
    oth = none & ~dup
    fr = E.decode_variant(f, v, 0.3, 0.6, False, 0.2)
    keep, c = fr["keep"], fr["rec"]
    s, d = c["src"].astype(int), c["dst"].astype(int)
    pn = np.asarray(f["pnone"], float)
    pn_dup.append(pn[dup])
    pn_other.append(pn[oth])
    n["frames"] += 1
    n["valid"] += int(v.sum())
    n["none_label"] += int(none.sum())
    n["dup"] += int(dup.sum())
    n["other_none"] += int(oth.sum())
    n["kept"] += int(keep.sum())
    n["kept_assigned"] += int((keep & ~none).sum())
    n["kept_dup"] += int((keep & dup).sum())
    n["kept_other_none"] += int((keep & oth).sum())
    td, to = dup[s] | dup[d], oth[s] | oth[d]
    n["links"] += len(s)
    n["links_touch_dup"] += int(td.sum())
    n["links_touch_other_none"] += int(to.sum())
    deg = np.bincount(s, minlength=len(keep)) + np.bincount(d, minlength=len(keep))
    jp = np.flatnonzero(deg >= 3)
    nd = set(s[td].tolist()) | set(d[td].tolist())
    no = set(s[to].tolist()) | set(d[to].tolist())
    n["deg3"] += len(jp)
    n["deg3_touch_dup"] += sum(int(x) in nd for x in jp)
    n["deg3_touch_other_none"] += sum(int(x) in no for x in jp)
    n["deg3_is_dup"] += int(dup[jp].sum())
pd_, po = np.concatenate(pn_dup), np.concatenate(pn_other)
out = dict(run=run, seed=seed, split=split, every=every, cell="short 0.3 / long 0.2 / keep 0.6 / gate off", counts=n,
           per_frame={k: v / max(n["frames"], 1) for k, v in n.items() if k != "frames"},
           shares=dict(dup_of_none=n["dup"] / max(n["none_label"], 1), dup_kept=n["kept_dup"] / max(n["dup"], 1), other_none_kept=n["kept_other_none"] / max(n["other_none"], 1),
                       kept_that_are_dup=n["kept_dup"] / max(n["kept"], 1), links_touch_dup=n["links_touch_dup"] / max(n["links"], 1), deg3_touch_dup=n["deg3_touch_dup"] / max(n["deg3"], 1),
                       deg3_touch_other_none=n["deg3_touch_other_none"] / max(n["deg3"], 1)),
           pnone_median=dict(dup=float(np.median(pd_)) if len(pd_) else None, other_none=float(np.median(po)) if len(po) else None),
           note="dup = valid, ytype none, and a GT successor of some source (y column non-empty); other_none = the other none-labelled valid tokens; deg3 = kept tokens with in + out degree >= 3 in the delivered token graph")
print(json.dumps(out, indent=1))
os.makedirs(os.path.join(ROOT, "temp", "e377", "out"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "temp", "e377", "out", f"dup_{run}_{split}_every{every}.json"), "w"), indent=1)
