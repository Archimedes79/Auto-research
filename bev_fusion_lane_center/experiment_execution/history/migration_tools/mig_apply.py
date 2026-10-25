"""Apply steps of the migration (moves only; every move is logged to MIGRATION_LOG.csv so it can be undone with mig_undo.py).
Usage (WSL, cwd anywhere):  python mig_apply.py A|B|C [--go]      without --go: print what would happen.
A  top-level folders into bev_fusion_lane_center (data, downloads, literature, reports, report_final, scripts, out/loop2 -> runs, ...)
B  scripts -> live / evaluation / tools / experiments code / archive / staged junk (plan.json)
C  run directories that live code does not need -> _to_delete/runs (plan.json)"""
import csv
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mig_common import *  # noqa: E402,F401,F403

step = sys.argv[1]
GO = "--go" in sys.argv
LOG = f"{RS}/MIGRATION_LOG.csv"
n_done = 0


def log(src, dst):
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "step", "src", "dst"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), step, src, dst])


def mv(src, dst):
    global n_done
    if not os.path.lexists(src):
        return
    if os.path.lexists(dst):
        print("  EXISTS, skipped:", dst)
        return
    n_done += 1
    if GO:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        log(src, dst)
    elif n_done <= 40:
        print("  would move", src.replace(O, "O"), "->", dst.replace(O, "O"))


plan = json.load(open(f"{RS}/plan.json"))

if step == "A":
    if GO:
        for d in ("", "literature", "experiment_execution", "experiments", "reports", "evaluation/lane_grid", "evaluation/sparse_lanes", "tools/viz", "tools/video", "tools/env", "tools/restructure", "scripts", "runs", "models", "data", "downloads", "_to_delete"):
            os.makedirs(f"{R}/{d}", exist_ok=True)
        # the log file moves into place
    top = [("data", "data"), ("downloads", "downloads"), ("literature", "literature"), ("reports", "reports"), ("report_final", "reports/source"), ("scripts", "scripts"), ("out/loop2", "runs"),
           ("v1.0-mini.tgz", "downloads/v1.0-mini.tgz"), ("nuScenes-map-expansion-v1.3.zip", "downloads/nuScenes-map-expansion-v1.3.zip"),
           ("bundles", "_to_delete/bundles"), ("mail13", "_to_delete/mail13"), ("recon", "_to_delete/recon"),
           ("out/crn", "runs/_loop1_crn"), ("out/flags", "_to_delete/out_flags"),
           ("recovered_loop17", "experiment_execution/loop17_recovery"),
           ("README.md", "experiment_execution/history/README_2026-10-13_loop1.md")]
    for s, d in top:
        # runs/ and scripts/ etc. exist as empty dirs created above -> move the *content*, so replace the empty dir
        src, dst = f"{O}/{s}", f"{R}/{d}"
        if GO and os.path.isdir(dst) and not os.listdir(dst):
            os.rmdir(dst)
        mv(src, dst)
    # loose files of out/ (old setup / crn logs) and of loop2/
    if os.path.isdir(f"{O}/out"):
        for f in sorted(os.listdir(f"{O}/out")):
            mv(f"{O}/out/{f}", f"{R}/_to_delete/out_logs/{f}")
    if os.path.isdir(f"{O}/loop2"):
        for f in sorted(os.listdir(f"{O}/loop2")):
            if f == "STATE.md":
                mv(f"{O}/loop2/{f}", f"{R}/experiment_execution/STATE.md")
            elif f == "STATE.md.bak_loop11":
                mv(f"{O}/loop2/{f}", f"{R}/experiment_execution/history/STATE_before_loop11.md")
            elif f == "splits.json":
                mv(f"{O}/loop2/{f}", f"{R}/data/splits.json")
            else:
                mv(f"{O}/loop2/{f}", f"{R}/scripts/_archive/reports_build/{f}")
elif step == "B":
    base = f"{R}/scripts"
    for name, dest in sorted(plan["scripts"].items()):
        if dest == "scripts":
            continue
        mv(f"{base}/{name}", f"{R}/{dest}/{name}")
elif step == "C":
    for nm, info in sorted(plan["runs"].items()):
        if not info["keep"]:
            mv(f"{R}/runs/{nm}", f"{R}/_to_delete/runs/{nm}")
print(f"step {step}: {n_done} moves {'done' if GO else 'planned (dry run)'}")
