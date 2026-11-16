"""one-off: gt.json + eval/metrics.json + eval/card.json (provenance block) of an E273 run from its scoring card, then the result package (package.py)

    wsl python3 temp/e273/finish_run.py e273_p_s0 E273.p_s0
"""
import hashlib
import json
import os
import subprocess
import sys

run, rid = sys.argv[1], sys.argv[2]
c = json.load(open(f"runs/e273/{run}/card.json"))
os.makedirs(f"runs/{run}/eval", exist_ok=True)
GT = {"version": "autogt/lane-l3", "signature": "3576fa1f7e"}                         # the lane GT; the front GT c643a7539d is not read by the head (the cached features were made with it)
json.dump({"gt_signature": "c643a7539d", "lane_signature": "3576fa1f7e", "gt_version": "front v8 (the cached features; not read by the head)", "lane_version": "autogt/lane-l3",
           "map_errors": "map_sim.make_map error mode, fixed per token"}, open(f"runs/{run}/gt.json", "w"), indent=1)
g, p = c["p"]["graph"], c["p"]
m = {"lane_acc_error_p": p["assign_error"]["lane_acc"], "lane_acc_error_g": c["g"]["assign_error"]["lane_acc"], "lane_acc_exact_p": p["assign_exact"]["lane_acc"], "lane_acc_exact_g": c["g"]["assign_exact"]["lane_acc"],
     "lane_acc_error_p_minus_g": c["diff"]["error"]["mean"], "lane_acc_exact_p_minus_g": c["diff"]["exact"]["mean"],
     **{k: g[k] for k in ("Q", "F1@1.0", "topo_F1", "junction_F1", "lanes_found", "dup_frac", "vert100_dp", "pieces_per_gt_piece") if k in g}, "node_F1": p["nodes"].get("node_F1")}
json.dump({k: float(v) for k, v in m.items() if v is not None}, open(f"runs/{run}/eval/metrics.json", "w"), indent=1)


def h(b):
    return hashlib.sha256(b).hexdigest()[:12]


ev_files = ["evaluation/qgraph_eval.py", "evaluation/qgraph_run_eval.py", "evaluation/lane_suite.py"]
ev_files = [f for f in ev_files if os.path.exists(f)]
code_hash = h(b"".join(open(f, "rb").read() for f in ev_files))
commit = subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = [ln[3:] for ln in subprocess.run(["git", "status", "--porcelain", "--"] + ev_files, capture_output=True, text=True).stdout.splitlines()]
rows = json.load(open("data/splits.json"))["val"]
prov = {"evaluator": {"commit": commit, "code_hash": code_hash, "dirty": dirty, "files": ev_files},
        "data": {"split": "val", "n": len(rows), "frame_hash": h(json.dumps([int(r) for r in rows]).encode()), "manifest_hash": "see provenance.json"},
        "gt": GT}
json.dump({"provenance": prov, "date": c["date"], "frames": c["frames"]}, open(f"runs/{run}/eval/card.json", "w"), indent=1)

pf = f"results/{rid}/provenance.json"
if os.path.exists(pf):                                                                  # the manifest was made before the GT names existed: add them, keep the rest
    pj = json.load(open(pf))
    pj["gt"] = GT | {"key": "lane-l3"}
    pj.setdefault("subject", {})["train_gt"] = dict(GT)                                 # package.py reads this one first (the manifest took the front GT of gt.json)
    pj["data"] = pj.get("data", {}) | {"split": "val", "n": len(rows), "frame_hash": prov["data"]["frame_hash"], "dataset": "nuScenes v1.0-trainval (validation scenes of data/splits.json)"}
    json.dump(pj, open(pf, "w"), indent=1)
r = subprocess.run(["python3", "../autoresearch/tools/package.py", "package", rid, "--run-dir", f"runs/{run}"], capture_output=True, text=True)
print((r.stdout + r.stderr)[-400:])
