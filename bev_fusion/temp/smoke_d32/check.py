"""one-off: steps d, f and g of the D32 smoke test after start.sh (run from bev_fusion/ with AR_PROJECT=../bev_fusion):
d trainsuite status of runs/smoke_d32; the GT signature of the card; f decide.py on the card against itself (must be TIE);
g ledger.py tail and schedule.py show"""
import glob
import json
import os
import subprocess
import sys

TOOLS = os.path.join("..", "autoresearch", "tools")


def sh(cmd, cwd="."):
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-2500:])
    return r


print("== d trainsuite status")
sh([sys.executable, "-m", "trainsuite", "--root", "../bev_fusion/runs", "status", "smoke_d32"], cwd="../autoresearch")

print("== e the card and its GT signature")
cards = sorted(glob.glob("runs/smoke_d32/eval/card*.json"))
print("cards:", cards)
card = json.load(open(cards[-1]))
gt = (card.get("protocol") or {}).get("gt_signature")
print("card protocol gt_signature:", gt, "| train gt.json:", (json.load(open("runs/smoke_d32/gt.json")) if os.path.exists("runs/smoke_d32/gt.json") else {}).get("gt_signature"))
head = "seg_fov/seg/dt-0.50/lane_divider/all/all/iou@0.5"  # the headline of the 6000-step run's rule
print("headline:", head, "frames:", card.get("frames"), "split:", card.get("split"))

print("== f decide.py: the card against itself")
v = card["metrics"][head]
print("headline value:", v)
r = sh([sys.executable, os.path.join(TOOLS, "decide.py"), "--variant", str(v), "--control", str(v)])
print("TIE as expected" if "TIE" in r.stdout else "NOT a TIE: look at the card")

print("== g ledger and schedule")
sh([sys.executable, os.path.join(TOOLS, "ledger.py"), "tail", "-n", "3"])
sh([sys.executable, os.path.join(TOOLS, "schedule.py"), "show"])
