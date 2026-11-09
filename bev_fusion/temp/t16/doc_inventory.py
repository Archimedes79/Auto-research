"""scratch: inventory of every tracked documentation file (.md, architecture drawings .svg / architecture json) of the repository: folder, size, last commit date, number of other files that name it
    wsl python3 bev_fusion/temp/t16/doc_inventory.py > bev_fusion/temp/t16/doc_inventory.txt   (from the repository root)"""
import collections
import os
import re
import subprocess

root = "."
files = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split("\n")
docs = [f for f in files if f and (f.endswith(".md") or (f.endswith(".svg") and "architecture" in f.lower()) or ("architecture" in os.path.basename(f).lower() and f.endswith(".json")))]
print("tracked files", len([f for f in files if f]), "docs", len(docs))
dates = {}
out = subprocess.run(["git", "log", "--format=%ad", "--date=short", "--name-only", "--", *[d for d in docs]], capture_output=True, text=True).stdout.split("\n")
cur = None
for ln in out:
    if re.match(r"^\d{4}-\d\d-\d\d$", ln):
        cur = ln
    elif ln and ln not in dates:
        dates[ln] = cur
# text of the other living files (for 'who names it'): code + docs, not history folders
texts = {}
for f in files:
    if f and f.endswith((".py", ".sh", ".md", ".json", ".csv", ".yml", ".ps1", ".txt")) and os.path.exists(f) and os.path.getsize(f) < 3_000_000:
        try:
            texts[f] = open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            pass
by_top = collections.defaultdict(list)
for d in docs:
    top = "/".join(d.split("/")[:2]) if d.count("/") >= 1 else "."
    by_top[top].append(d)
print()
for top, ds in sorted(by_top.items(), key=lambda kv: -len(kv[1])):
    print(f"{top}: {len(ds)} files, {sum(os.path.getsize(d) for d in ds if os.path.exists(d)) // 1024} KB")
print()
rows = []
for d in docs:
    if not os.path.exists(d):
        continue
    base = os.path.basename(d)
    refs = [f for f, t in texts.items() if f != d and (d in t or (base not in ("README.md", "experiment.md", "CLAUDE.md") and base in t))]
    code_refs = [f for f in refs if f.endswith((".py", ".sh", ".json", ".ps1"))]
    rows.append((d, os.path.getsize(d) // 1024, dates.get(d, "?"), len(refs), len(code_refs)))
print("path | KB | last commit | named by n files | of them code/json")
for r in sorted(rows):
    print(" | ".join(str(x) for x in r))
