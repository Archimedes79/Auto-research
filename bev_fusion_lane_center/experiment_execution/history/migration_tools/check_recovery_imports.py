"""Which local modules do the replayed loop-17 scripts import that are not in the live folders (they may sit in scripts/_archive)?"""
import ast
import os

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
REC = f"{R}/experiment_execution/loop17_recovery/tree/scripts"
live = set()
for d in (f"{R}/scripts", f"{R}/evaluation/lane_grid", f"{R}/evaluation/sparse_lanes", f"{R}/tools/viz", f"{R}/tools/video"):
    live |= {f[:-3] for f in os.listdir(d) if f.endswith(".py")}
rec = {f[:-3] for f in os.listdir(REC) if f.endswith(".py")}
where = {}
for dp, dn, fn in os.walk(f"{R}/scripts"):
    for f in fn:
        if f.endswith(".py"):
            where.setdefault(f[:-3], os.path.relpath(dp, R))
need = {}
for f in sorted(rec):
    try:
        tree = ast.parse(open(f"{REC}/{f}.py", encoding="utf-8-sig").read())
    except SyntaxError as e:
        print("SYNTAX", f, e)
        continue
    for nd in ast.walk(tree):
        mods = []
        if isinstance(nd, ast.Import):
            mods = [a.name.split(".")[0] for a in nd.names]
        elif isinstance(nd, ast.ImportFrom) and nd.module:
            mods = [nd.module.split(".")[0]]
        for m in mods:
            if m in rec or m in live:
                continue
            if m in where:
                need.setdefault(m, set()).add(f)
print("recovered scripts:", len(rec))
print("modules to bring back from the archive:", {m: (where[m], sorted(v)[:3]) for m, v in need.items()})
