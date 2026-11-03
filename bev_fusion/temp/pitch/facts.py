"""Facts of the repository for the case of 2026-11-01 (stdlib only; every number comes from a file or from git, nothing typed).
Run from the repository root: python3 bev_fusion/temp/pitch/facts.py > bev_fusion/temp/pitch/facts.json"""
import csv
import collections
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
LANE = os.path.join(ROOT, "bev_fusion_lane_center")
BEV = os.path.join(ROOT, "bev_fusion")


def git(*a):
    return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout


def ledger(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    out = {"rows": len(rows), "status": dict(collections.Counter(r["status"] for r in rows))}
    secs = [float(r["duration_s"]) for r in rows if r["duration_s"].strip()]
    out["wall_hours"] = round(sum(secs) / 3600, 1)
    out["with_prediction"] = sum(1 for r in rows if r["prediction"].strip())
    out["with_rule"] = sum(1 for r in rows if r["rule"].strip())
    out["with_decision"] = sum(1 for r in rows if r["decision"].strip())
    return out


facts = {}
facts["lane_ledger"] = ledger(os.path.join(LANE, "experiment_execution", "LEDGER.csv"))
facts["bev_ledger"] = ledger(os.path.join(BEV, "records", "LEDGER.csv"))

# decisions
dec = open(os.path.join(LANE, "experiment_execution", "DECISIONS.md"), encoding="utf-8").read()
ids = sorted({int(m) for m in re.findall(r"^\| D(\d+)\b", dec, re.M)})
open_sec = dec.split("## Open")[1].split("## Decided")[0]
open_ids = sorted({int(m) for m in re.findall(r"^\| D(\d+)\b", open_sec, re.M)})
facts["decisions"] = {"ids_seen": len(ids), "highest": max(ids), "open": [f"D{i}" for i in open_ids]}

# git
log = git("log", "--format=%ad", "--date=short").split()
per_day = collections.Counter(log)
facts["git"] = {"commits": len(log), "first_day": min(log), "last_day": max(log), "per_day": dict(sorted(per_day.items())),
                "tags": dict(collections.Counter(t.split("/")[0] for t in git("tag").split()))}

# experiments, literature
exp_lane = [d for d in os.listdir(os.path.join(LANE, "experiments")) if re.match(r"E\d+", d)]
exp_bev = [d for d in os.listdir(os.path.join(BEV, "experiments")) if re.match(r"t\d+", d)]
facts["experiments"] = {"lane_folders": len(exp_lane), "lane_highest_E": max(int(re.match(r"E(\d+)", d).group(1)) for d in exp_lane),
                        "bev_folders": len(exp_bev)}
lit = os.path.join(LANE, "literature")
facts["literature"] = {"papers_csv_rows": sum(1 for _ in open(os.path.join(lit, "papers.csv"), encoding="utf-8")) - 1,
                       "paper_notes": len(os.listdir(os.path.join(lit, "papers"))), "deep_notes": len(os.listdir(os.path.join(lit, "deep")))}
facts["agents"] = len([f for f in os.listdir(os.path.join(ROOT, "autoresearch", "agents")) if f.endswith(".md")])
facts["tools"] = len([f for f in os.listdir(os.path.join(ROOT, "autoresearch", "tools")) if f.endswith(".py")])

json.dump(facts, sys.stdout, indent=1)
print()
