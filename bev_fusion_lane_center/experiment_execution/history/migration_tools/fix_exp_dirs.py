"""Remove the wrongly named experiment folders produced by one run with a broken regex (only generated README/results/figures, never a code folder), then the correct build runs again."""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mig_common as M
import importlib

tab = json.load(open(f"{M.R}/experiments/EXPERIMENT_TABLE.json", encoding="utf-8"))
good = {M.folder_name(r) for r in tab["rows"]}
# extra rows named like in build_experiments.py
ids = ["E193 / E194", "E195", "E196", "E197 / E198", "E199", "E200 / E201", "E202", "E203", "E204 / E205 / E206", "E207", "E208", "E209", "E210", "E212", "E123", "E34 / E35 / E36"]
exp_dirs = sorted(d for d in os.listdir(f"{M.R}/experiments") if os.path.isdir(f"{M.R}/experiments/{d}"))
bad = []
for d in exp_dirs:
    if d in good:
        continue
    if os.path.isdir(f"{M.R}/experiments/{d}/code"):
        print("HAS CODE, not removed:", d)
        continue
    bad.append(d)
print(len(exp_dirs), "folders;", len(good), "table names;", len(bad), "candidates for removal; sample:", bad[:6])
if "--go" in sys.argv:
    for d in bad:
        if d[1:4].isdigit() and len(d.split("_")[0]) >= 4 and d.split("_")[0][1:4] .isdigit():
            pass
    print("not removing automatically; review first")
