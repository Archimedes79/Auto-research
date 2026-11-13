"""Import closure of the live head vs the rest (scratch): which files does the current head (E273 - E319) import, which only the old prototype chain, which nobody."""
import ast
import collections
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
files = {}
for d in ("training", "evaluation", "tools"):
    for dp, dn, fn in os.walk(os.path.join(ROOT, d)):
        p = dp.replace("\\", "/")
        if "__pycache__" in p:
            continue
        for f in fn:
            if f.endswith(".py"):
                files.setdefault(f[:-3], []).append(os.path.join(dp, f))


def imports(path):
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[-1] for a in n.names} | {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                out |= {n.module.split(".")[-1], n.module.split(".")[0]}
            out |= {a.name for a in n.names}
    return out


def closure(entries):
    seen, todo = set(), list(entries)
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        for m in imports(p):
            for q in files.get(m, []):
                todo.append(q)
    return seen


def find(name):
    return [q for q in files.get(name.replace(".py", "").split("/")[-1], [])]


active_names = ["qgraph_train", "jends_train", "jmaps_train", "run_config", "grid_train", "startable", "trainer", "qgraph_run_eval", "jends_eval", "jmaps_eval", "qends_eval", "jmaneuver", "jdiag",
                "jpair_pr", "jrule_control", "jsplits", "readout_replay", "qgraph_eval", "qgraph_timing", "vertex_eval", "lane_suite", "qgraph_view", "junction_view", "qgraph_results", "qgraph_readout"]
active_entries = [q for n in active_names for q in find(n)]
active = closure(active_entries)
# every file named in the test lists
listed = set()
for sh in ("tools/maintenance/run_all_tests.sh", "tools/maintenance/run_gates.sh"):
    for ln in open(os.path.join(ROOT, sh), encoding="utf-8"):
        m = re.search(r'^run\s+"[^"]*"\s+(\S+\.py)', ln)
        if m:
            listed |= set(find(m.group(1)))
alltests = closure(list(listed) + active_entries)
# the active head's own tests count as active
active_tests = {q for q in listed if any(os.path.basename(q)[:-3].replace("test_", "") in os.path.basename(a) for a in active)}


def lines(p):
    return sum(1 for _ in open(p, encoding="utf-8", errors="replace"))


def group(paths, title):
    g = collections.defaultdict(lambda: [0, 0])
    for p in paths:
        k = os.path.relpath(os.path.dirname(p), ROOT).replace("\\", "/")
        g[k][0] += 1
        g[k][1] += lines(p)
    print(f"\n{title}: {sum(v[0] for v in g.values())} files, {sum(v[1] for v in g.values())} lines")
    for k, v in sorted(g.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"  {v[1]:6d} lines {v[0]:3d} files  {k}")


allp = {p for v in files.values() for p in v}
group(active, "imported by the live head (entry points + closure)")
group(alltests - active, "kept alive only by the full test list / old chain")
group(allp - alltests - active, "imported by nobody and run by no test (orphans)")
print("\nlargest orphans:")
for p in sorted(allp - alltests - active, key=lambda p: -lines(p))[:15]:
    print(f"  {lines(p):5d} {os.path.relpath(p, ROOT)}")
print("\nlargest files only the old chain keeps alive:")
for p in sorted(alltests - active, key=lambda p: -lines(p))[:15]:
    print(f"  {lines(p):5d} {os.path.relpath(p, ROOT)}")
