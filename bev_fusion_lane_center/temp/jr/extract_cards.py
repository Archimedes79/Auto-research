"""one-off: pull id, title, question, verdict lines and the ledger decision out of the cards E273 - E330 (scratch for the total report).
    wsl python3 temp/jr/extract_cards.py > temp/jr/cards_summary.json   (run from bev_fusion_lane_center)"""
import csv
import glob
import json
import os
import re
import sys

ROOT = os.getcwd()
out = []
led = {}
for r in csv.DictReader(open(os.path.join(ROOT, "experiment_execution", "LEDGER.csv"), encoding="utf-8")):
    led.setdefault(r["id"].split(".")[0], []).append({"id": r["id"], "status": r["status"], "decision": r["decision"][:260], "result": r["result"][:260]})
for d in sorted(glob.glob(os.path.join(ROOT, "experiments", "E*")), key=lambda p: os.path.basename(p)):
    name = os.path.basename(d)
    m = re.match(r"E(\d+)([a-z]?)_(.*)", name)
    if not m or not (273 <= int(m.group(1)) <= 340):
        continue
    f = os.path.join(d, "experiment.md")
    if not os.path.exists(f):
        continue
    t = open(f, encoding="utf-8").read()
    h1 = t.split("\n", 1)[0].lstrip("# ").strip()
    para = ""
    for blk in t.split("\n\n")[1:]:
        if blk.strip() and not blk.startswith("#") and not blk.startswith("```"):
            para = blk.strip()
            break
    card = None
    mm = re.search(r"```json card\n(.*?)\n```", t, re.S)
    if mm:
        try:
            card = json.loads(mm.group(1))
        except ValueError:
            card = None
    verd = [ln.strip() for ln in t.split("\n") if re.search(r"\*\*Verdict|Verdict by the rule|^\*\*(KEEP|TIE|CUT)|Decision:|Result \(", ln)][:6]
    out.append({"id": "E" + m.group(1) + m.group(2), "slug": m.group(3), "h1": h1, "question": para[:700],
                "title": card.get("title") if card else None, "rule": card.get("rule") if card else None, "seeds": card.get("seeds") if card else None,
                "budget": card.get("budget_gpu_h") if card else None, "verdict_lines": verd, "ledger": led.get("E" + m.group(1) + m.group(2), [])[:4], "chars": len(t)})
json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
