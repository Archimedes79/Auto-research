"""Tighten the live set: modules of scripts/ that only historic (archived) scripts need go to scripts/_archive. Prints; --go moves."""
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mig_common import *  # noqa: E402,F401,F403
import calendar
import time

GO = "--go" in sys.argv
SC = f"{R}/scripts"
INV = {x["name"]: x["mtime"] for x in json.load(open(f"{RS}/inventory.json"))["scripts"]}  # mtimes BEFORE the path rewrite touched every file
RECENT = calendar.timegm(time.strptime("2026-10-22", "%Y-%m-%d"))
ROOT_PAT = re.compile(r"^(p1[3-7]_|e2e_|loop1[4-7]|build_loop1[4-7]|fig_p1[4-7]|graph_eval|report_lib|arch_figs)")
import ast

top = [f for f in os.listdir(SC) if os.path.isfile(f"{SC}/{f}") and f.endswith(".py")]
mods = {f[:-3] for f in top}
extra = {"graph_eval", "viz_graph"}  # live modules that moved to evaluation/ and tools/viz
imports = {}
for f in top:
    try:
        tree = ast.parse(open(f"{SC}/{f}", encoding="utf-8-sig", errors="replace").read())
    except SyntaxError:
        imports[f[:-3]] = set()
        continue
    s = set()
    for nd in ast.walk(tree):
        if isinstance(nd, ast.Import):
            s.update(a.name.split(".")[0] for a in nd.names)
        elif isinstance(nd, ast.ImportFrom) and nd.module:
            s.add(nd.module.split(".")[0])
    imports[f[:-3]] = s & (mods | extra)
roots = [m for m in mods if ROOT_PAT.match(m) or INV.get(m + ".py", 0) >= RECENT]
live, todo = set(), list(roots)
while todo:
    m = todo.pop()
    if m in live or m not in mods:
        continue
    live.add(m)
    todo.extend(imports.get(m, []))
# shell wrappers pull in the scripts they name
for f in os.listdir(SC):
    if f.endswith((".sh", ".ps1")):
        for m in re.findall(r"\b([A-Za-z0-9_\-]+)\.py\b", open(f"{SC}/{f}", encoding="utf-8", errors="replace").read()):
            if m in mods and m not in live:
                todo.append(m)
                while todo:
                    x = todo.pop()
                    if x in live or x not in mods:
                        continue
                    live.add(x)
                    todo.extend(imports.get(x, []))
dead = sorted(m for m in mods if m not in live and m not in ("gcar_tools", "centerline_gt"))
print(f"modules in scripts/: {len(mods)}, live: {len(live)}, to archive: {len(dead)}")
print(" ".join(dead))
if GO:
    for m in dead:
        g = archive_group(m + ".py")
        dst = f"{SC}/_archive/{'tools_env' if g == 'tools_env' else g}"
        os.makedirs(dst, exist_ok=True)
        shutil.move(f"{SC}/{m}.py", f"{dst}/{m}.py")
        with open(f"{RS}/MIGRATION_LOG.csv", "a", encoding="utf-8") as lg:
            lg.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},live2,{SC}/{m}.py,{dst}/{m}.py\n")
    print("moved", len(dead))
