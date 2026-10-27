"""One-off: short smoke versions (50 steps, out_root temp/train/runs) of t2 configs: python3 temp/study/smoke_cfgs.py arm [arm ...]"""
import json, os, sys
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for a in sys.argv[1:]:
    c = json.load(open(os.path.join(root, f"training/configs/t2_{a}.json")))
    c.update({"name": f"t2s_{a}", "steps": 50, "out_root": "temp/train/runs", "eval_every": 1000, "save_every": 1000, "log_every": 10})
    json.dump(c, open(os.path.join(root, f"temp/study/t2s_{a}.json"), "w"), indent=1)
