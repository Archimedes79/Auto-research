"""audit scratch: unused imports (module level and in functions) and assigned-but-never-read local variables in the scope files."""
import ast
import os
import sys

TR = "/mnt/c/Auto-research/bev_fusion/training"
SCOPE = "model losses loss_extras layers interface fusion bev_trunk blocks_new fallbacks radar_bev radar_encoder splat temporal temporal_fusion motion_block object_head timed_head extract lane_branch lift_pd bevfusion_camera camera_encoder launch_saving".split()
files = [os.path.join(TR, m + ".py") for m in SCOPE] + [os.path.join(TR, "legacy", f) for f in sorted(os.listdir(os.path.join(TR, "legacy"))) if f.endswith(".py")]
for path in files:
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    used = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            used.add(n.id)
        elif isinstance(n, ast.Attribute):
            pass
    # names in __all__ / strings are ignored; noqa lines are reported separately
    lines = src.split("\n")
    imported = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                imported.append(((a.asname or a.name).split(".")[0], n.lineno))
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                imported.append((a.asname or a.name, n.lineno))
    seen = set()
    for name, ln in imported:
        if name in used or name == "*":
            continue
        # a name used only in a string/doc is still unused
        key = (name, ln)
        if key in seen:
            continue
        seen.add(key)
        noqa = "noqa" in lines[ln - 1]
        print(f"UNUSED-IMPORT {os.path.relpath(path, TR)}:{ln} {name}{'  [noqa]' if noqa else ''}")
    # assigned-but-unused locals per function
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        stores, loads = {}, set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Name):
                if isinstance(n.ctx, ast.Store):
                    stores.setdefault(n.id, n.lineno)
                else:
                    loads.add(n.id)
        glob = {n for n in ast.walk(fn) if isinstance(n, (ast.Global, ast.Nonlocal))}
        if glob:
            continue
        for name, ln in stores.items():
            if name not in loads and not name.startswith("_") and name != "self":
                print(f"UNUSED-LOCAL  {os.path.relpath(path, TR)}:{ln} {name} in {fn.name}")
