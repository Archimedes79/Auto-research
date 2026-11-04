"""throw-away, read-only: tracked .md files of bev_fusion / autoresearch / autogt / datasets / top level, ranked by dead path references (backticked paths that no longer exist,
resolved against the file's folder, the repository root and the project folders) and by the age of the last commit; phrases that mark a text as waiting ("waiting for", "not yet", "TODO")"""
import os
import re
import subprocess
import time

REPO = "/mnt/c/Auto-research"
run = lambda *a: subprocess.run(a, capture_output=True, cwd=REPO).stdout.decode("utf-8", "replace")
files = [f for f in run("git", "ls-files", "*.md").splitlines() if f.split("/")[0] in ("bev_fusion", "autoresearch", "autogt", "datasets") or "/" not in f]
files = [f for f in files if "/history/" not in f and "/literature/" not in f]
last = {}
cur = None
for ln in run("git", "log", "--format=@@%cs", "--name-only", "--", "*.md").splitlines():
    if ln.startswith("@@"):
        cur = ln[2:]
    elif ln and ln not in last:
        last[ln] = cur
PATH = re.compile(r"`((?:\.\./)*[A-Za-z0-9_./-]+\.(?:py|md|json|csv|sh|txt|svg|pt))`")
WAIT = re.compile(r"waiting for|not yet|TODO|to be decided|open point", re.I)
bases = ["", "bev_fusion", "bev_fusion/training", "bev_fusion/evaluation", "bev_fusion/tests", "autoresearch", "autoresearch/tools", "autogt"]
rows = []
for f in files:
    p = os.path.join(REPO, f)
    try:
        txt = open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        continue
    dead, scratch = [], 0
    for m in set(PATH.findall(txt)):
        if m.startswith("data/") or m.startswith("runs/") or "/runs/" in m or m.startswith("temp/") or "/temp/" in m or "_to_delete" in m or m.startswith("results/"):
            scratch += 1
            continue
        cands = [os.path.normpath(os.path.join(REPO, os.path.dirname(f), m))] + [os.path.normpath(os.path.join(REPO, b, m)) for b in bases]
        if not any(os.path.exists(c) for c in cands):
            dead.append(m)
    rows.append((len(dead), len(WAIT.findall(txt)), last.get(f, "?"), len(txt) // 1024, f, sorted(dead)[:3]))
rows.sort(key=lambda r: (-r[0], r[2]))
print(f"{len(rows)} tracked md files scanned (history/ and literature/ left out)")
print("dead refs | waiting phrases | last commit | KB | file | first dead refs")
for r in rows[:22]:
    print(f"{r[0]:3d} | {r[1]:3d} | {r[2]} | {r[3]:4d} | {r[4]} | {', '.join(r[5])}")
old = sorted((r for r in rows if r[0] == 0 and r[2] != "?"), key=lambda r: r[2])[:6]
print("oldest without dead refs:")
for r in old:
    print(f"      {r[2]} | {r[3]:4d} KB | {r[4]}")
