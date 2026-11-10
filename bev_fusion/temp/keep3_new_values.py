"""Scratch: the board7 of every arm on the new GT (v8, 0.5_4_6a7aa7864a), computed the way keep3.py info_arm does (mean over the seeds of the mean of the seven keys; every seed has the keys, the split val and the signature)"""
import glob
import json
import os
import re
from collections import defaultdict

ROOT = "/mnt/c/Auto-research/bev_fusion"
k3 = json.load(open(os.path.join(ROOT, "records", "keep3.json")))
keys, SIG = k3["metric_keys"], "0.5_4_6a7aa7864a"
arms = defaultdict(list)
for d in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "eval", "metrics.json"))):
    run = d.split("/")[-3]
    arms[re.sub(r"_s\d+$", "", run)].append((run, d))
rows = []
for arm, seeds in arms.items():
    vals, why, ckpts = [], "", []
    for run, cp in seeds:
        c = json.load(open(cp, encoding="utf-8-sig"))
        m = c.get("metrics", {})
        miss = [x for x in keys if not isinstance(m.get(x), (int, float)) or m[x] != m[x]]
        cks = sorted(glob.glob(os.path.join(ROOT, "runs", run, "ckpt_*.pt")))
        if miss:
            why = f"{run}: {len(miss)} keys missing"; break
        if c.get("split") != "val" or c.get("gt_signature") != SIG:
            why = f"{run}: split {c.get('split')} GT {c.get('gt_signature')}"; break
        if not cks or (c.get("checkpoint") and os.path.basename(cks[-1]) != c["checkpoint"]):
            why = f"{run}: checkpoint {c.get('checkpoint')} vs {[os.path.basename(x) for x in cks][-1:]}"; break
        vals.append(sum(float(m[x]) for x in keys) / len(keys))
        ckpts.append(os.path.basename(cks[-1]))
    rows.append((arm, len(seeds), sum(vals) / len(vals) if not why and vals else None, why, ckpts, (c.get("latency") or {})))
print(f"{'arm':26} seeds  board7   note")
for arm, n, v, why, ck, lat in sorted(rows, key=lambda r: -(r[2] if r[2] is not None else -1)):
    if v is not None:
        print(f"{arm:26} {n:3d}   {v:.4f}   ckpt {ck[0] if ck else ''}")
print("--- not rankable on this GT:")
for arm, n, v, why, ck, lat in rows:
    if v is None:
        print(f"  {arm:26} {why[:110]}")
