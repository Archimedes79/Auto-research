"""one-off: training/README.md of HEAD + only my two changes (the extract.py row, the architecture_target row wording):
python readme_rows.py <head copy> <working file> <out>"""
import sys

head = open(sys.argv[1], encoding="utf-8").read().split("\n")
work = open(sys.argv[2], encoding="utf-8").read().split("\n")
mine = {l.split(" | ")[0]: l for l in work if l.startswith("| `extract.py` |") or l.startswith("| `architecture_target.json` /")}
assert len(mine) == 2, mine.keys()
out = []
for l in head:
    key = l.split(" | ")[0]
    if key in mine and l.startswith("| `architecture_target.json` /"):
        out.append(mine[key])
        continue
    if l.startswith("| `bev_trunk.py` |"):
        out.append(mine["| `extract.py`"])
    out.append(l)
assert sum(l.startswith("| `extract.py` |") for l in out) == 1
open(sys.argv[3], "w", encoding="utf-8", newline="").write("\n".join(out))
print("README: HEAD + 2 rows")
