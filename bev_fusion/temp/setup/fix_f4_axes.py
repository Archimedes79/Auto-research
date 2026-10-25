"""One-off: the F4 scouts wrote axis 'F4', which catalogue.py does not know; map each scout table to a valid axis.

lift / radar / heads / sdf -> A1 (sensing foundation), temporal -> A5; an extract that does not start with
deep / skip gets the prefix 'skip: '. Run from the problem folder: wsl python3 temp/loop19/fix_f4_axes.py
"""
import csv
import os

LIT = "literature"
AXIS = {"lift": "A1", "radar": "A1", "heads": "A1", "sdf": "A1", "temporal": "A5"}
target = {}
for topic, axis in AXIS.items():
    with open(os.path.join(LIT, f"papers_new_{topic}.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            target.setdefault(r["id"].strip(), axis)

path = os.path.join(LIT, "papers.csv")
with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fields, rows = reader.fieldnames, list(reader)
changed = 0
for r in rows:
    if r["axis"] == "F4" and r["id"] in target:
        r["axis"] = target[r["id"]]
        changed += 1
        ext = r["extract"].strip()
        if ext and not ext.startswith(("deep", "skip")):
            r["extract"] = "skip: " + ext
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\r\n")
    w.writeheader()
    w.writerows(rows)
print("axis changed:", changed, "left F4:", sum(r["axis"] == "F4" for r in rows))
