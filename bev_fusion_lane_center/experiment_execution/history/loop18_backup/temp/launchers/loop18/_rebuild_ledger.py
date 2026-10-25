"""Rebuild experiment_execution/LEDGER.csv from the run records (run.json) after the ledger was truncated on 2026-10-25 00:01 (an old run.py met the new ledger columns).
Rows are written with the writer of tools/experiment/run.py (current columns); rows of the ledger that are not in a run record are kept. Usage (any python of the WSL): python3 _rebuild_ledger.py"""
import csv
import glob
import importlib.util
import json
import os

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
spec = importlib.util.spec_from_file_location("run_mod", f"{R}/tools/experiment/run.py")
run_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_mod)
recs = []
for p in glob.glob(f"{R}/experiment_execution/runs/*/run.json"):
    try:
        recs.append(json.load(open(p)))
    except Exception as e:
        print("unreadable", p, e)
recs.sort(key=lambda r: r.get("started", ""))
for r in recs:
    res = ""
    row = dict(r, result=res, notes=(f"{r.get('attempts', 1)} attempt(s)" if r.get("attempts", 1) > 1 else ""), prediction=r.get("prediction", r.get("pred", "")), rule=r.get("rule", ""))
    run_mod.ledger_write(row)
print("rows written:", len(recs))
print(open(run_mod.LEDGER, encoding="utf-8").read()[:600])
