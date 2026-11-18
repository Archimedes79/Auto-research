"""E363 stop rule (ii'/iv): the surgery and the dumps, as arrays. For every run folder `runs/e363_<g>_s<S>` (g = c2, a, b): (1) `nodes.*` of its model.pt equal those of the node-head source (`runs/e292_<g>_s<S>` or, for b, `runs/e273_p_s<S>`), (2) every OTHER tensor equals `runs/e324_s_s<S>/model.pt`,
(3) provenance.json: the sha256 of model.pt equals the file's, (4) the arrays that do not depend on the node head (kp_pos, kp_dir, kp_p, logit, valid, allowed, scale) of its dump equal the stored `runs/e324_s_s<S>/ends_val.pkl` BIT FOR BIT on all 873 frames, (5) for b: every array of the dump (att_* included) equals the stored dump bit for bit (the FIXTURE).
Writes temp/e363/surgery_check.json (`pass`, details).   usage: wsl bash tools/env/_py14.sh surgery_check.py"""
import hashlib
import json
import os
import pickle
import numpy as np
import torch

ROOT = os.environ["BFLC_ROOT"]
res = dict(runs={}, pass_=True)
NODE_FREE = ("kp_pos", "kp_dir", "kp_p", "logit", "valid", "allowed", "scale")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731


def same(u, v):
    if u is None or v is None:
        return (u is None) == (v is None)
    return np.array_equal(np.asarray(u), np.asarray(v), equal_nan=True)


for g in ("c2", "a", "b"):
    for s in (0, 1):
        run = f"e363_{g}_s{s}"
        d = f"{ROOT}/runs/{run}"
        if not os.path.exists(f"{d}/model.pt"):
            continue
        r = dict(ok=True, notes=[])
        mine = torch.load(f"{d}/model.pt", map_location="cpu")["state"]
        src = torch.load(f"{ROOT}/runs/e324_s_s{s}/model.pt", map_location="cpu")["state"]
        nsrc = torch.load(f"{ROOT}/runs/" + (f"e273_p_s{s}" if g == "b" else f"e292_{g}_s{s}") + "/model.pt", map_location="cpu")["state"]
        for k, v in mine.items():
            ref = nsrc[k] if k.startswith("nodes.") else src[k]
            if not torch.equal(v, ref):
                r["ok"] = False
                r["notes"].append(f"tensor {k} differs from its source")
        r["notes"].append(f"{sum(k.startswith('nodes.') for k in mine)} node tensors checked against the node-head source, {sum(not k.startswith('nodes.') for k in mine)} others against E324 s")
        pv = json.load(open(f"{d}/provenance.json"))
        if pv["sha256_new"] != sha(f"{d}/model.pt"):
            r["ok"] = False
            r["notes"].append("provenance sha256 differs from model.pt")
        dp = f"{d}/ends_val.pkl"
        if os.path.exists(dp):
            new = pickle.load(open(dp, "rb"))
            old = pickle.load(open(f"{ROOT}/runs/e324_s_s{s}/ends_val.pkl", "rb"))
            keys = [k for k in old[0] if k not in ("row", "token")] if g == "b" else NODE_FREE
            bad = [(x["row"], k) for x, y in zip(old, new) for k in keys if (x["row"] != y["row"]) or not same(x.get(k), y.get(k))]
            if len(new) != 873 or len(old) != 873 or bad:
                r["ok"] = False
                r["notes"].append(f"dump: {len(new)} frames, {len(bad)} differing arrays {bad[:3]}")
            r["notes"].append(f"dump compared on {len(keys)} fields over {len(new)} frames: {'bit for bit' if not bad else 'DIFFERENT'}")
        else:
            r["notes"].append("no dump yet")
            r["ok"] = r["ok"] and g != "b" and False
        res["runs"][run] = r
        res["pass_"] = res["pass_"] and r["ok"]
        print(run, "OK" if r["ok"] else "FAIL", "; ".join(r["notes"])[:300])
res["pass"] = bool(res.pop("pass_"))
json.dump(res, open(f"{ROOT}/temp/e363/surgery_check.json", "w"), indent=1)
print("SURGERY CHECK", "PASS" if res["pass"] else "FAIL")
