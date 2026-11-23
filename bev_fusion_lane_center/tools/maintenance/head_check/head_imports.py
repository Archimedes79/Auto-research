"""The clean-copy check (tools/maintenance/head_check/head_check.sh), part 2: the import audit; runs INSIDE a git-archive copy of HEAD (BFLC_ROOT = its problem folder, PYTHONPATH from its own
tools/env/bflc_env.sh). Every import statement (also inside functions) of every committed .py of the problem folder and autoresearch/ is resolved: to a .py / package
of the copy (the PYTHONPATH dirs, the file's own folder, any other folder of the copy = a sys.path insert), to the gcar environment (stdlib, site-packages), or NOT;
an unresolved name is looked up in the LIVE tree's code folders: there = a file the commit lacks (the case to catch before the push), nowhere = an optional or dead import
(a guarded one is marked). Also writes the list of README-named modules (temp/head_named.txt) for the real imports (`head_import1.py`, one process each).
usage: _py14.sh <copy>/bev_fusion_lane_center/temp/head_imports.py <live repo root>   -> prints the summary, writes temp/head_imports.json"""
import ast
import importlib.util
import json
import os
import re
import sys

C = os.environ["BFLC_ROOT"]
REPO = os.path.dirname(C)
LIVE = sys.argv[1]
skip_dir = lambda d: "__pycache__" in d or "/.git" in d  # noqa: E731

pyfiles = [os.path.join(d, f) for top in (C, os.path.join(REPO, "autoresearch")) for d, _, fs in os.walk(top) if not skip_dir(d) for f in fs if f.endswith(".py")]
pyfiles = [p for p in pyfiles if "/temp/head_import" not in p]
path_dirs = [p for p in os.environ.get("PYTHONPATH", "").split(":") if p]
all_dirs = sorted({d for d, _, fs in os.walk(REPO) if not skip_dir(d)})


def provides(d, name):
    return os.path.isfile(os.path.join(d, name + ".py")) or os.path.isfile(os.path.join(d, name, "__init__.py")) or os.path.isdir(os.path.join(d, name)) and any(f.endswith(".py") for f in os.listdir(os.path.join(d, name)))


env_cache = {}


def in_env(name):
    if name not in env_cache:
        try:
            spec = importlib.util.find_spec(name)
            env_cache[name] = spec is not None and not (spec.origin or "").startswith(REPO)
        except Exception:
            env_cache[name] = False
    return env_cache[name]


live_index = {}                                                              # the code folders of the live tree only (runs / data / caches are not walked)
LP = os.path.join(LIVE, "bev_fusion_lane_center")
for top in [os.path.join(LP, x) for x in ("training", "evaluation", "tools", "reports/build", "temp")] + [os.path.join(LIVE, x) for x in ("autoresearch", "GtGenerator", "temp")]:
    for d, ds, fs in os.walk(top):
        ds[:] = [x for x in ds if x not in ("__pycache__", ".git", "runs", "data", "out", "cache")]
        for f in fs:
            if f.endswith(".py"):
                live_index.setdefault(f[:-3], []).append(os.path.relpath(os.path.join(d, f), LIVE))


def guarded(tree, node):
    for t in ast.walk(tree):
        if isinstance(t, ast.Try) and any(node in ast.walk(s) for s in t.body):
            for h in t.handlers:
                names = [] if h.type is None else [getattr(x, "id", getattr(x, "attr", "")) for x in (h.type.elts if isinstance(h.type, ast.Tuple) else [h.type])]
                if h.type is None or set(names) & {"ImportError", "ModuleNotFoundError", "Exception"}:
                    return True
    return False


unresolved, via_repo, parse_err = [], [], []
for p in pyfiles:
    try:
        tree = ast.parse(open(p, encoding="utf-8").read())
    except Exception as e:
        parse_err.append((os.path.relpath(p, REPO), repr(e)[:120]))
        continue
    own = os.path.dirname(p)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names = [node.module.split(".")[0]]
        else:
            continue
        for n in names:
            if provides(own, n) or any(provides(d, n) for d in path_dirs) or in_env(n):
                continue
            if any(provides(d, n) for d in all_dirs):
                via_repo.append((os.path.relpath(p, REPO), n))
                continue
            unresolved.append(dict(file=os.path.relpath(p, REPO), line=node.lineno, name=n, guarded=guarded(tree, node), live=live_index.get(n, [])))

named, missing = set(), set()
for d, _, fs in os.walk(C):
    if skip_dir(d) or "README.md" not in fs:
        continue
    for m in re.findall(r"`([A-Za-z0-9_./-]+\.py)`", open(os.path.join(d, "README.md"), encoding="utf-8").read()):
        cands = [os.path.join(d, m)] + [os.path.join(x, m) for x in path_dirs]
        hit = next((c for c in cands if os.path.isfile(c)), None)
        if hit is None:
            hits = sorted({os.path.join(x, os.path.basename(m)) for x in path_dirs if os.path.isfile(os.path.join(x, os.path.basename(m)))})
            hit = hits[0] if len(hits) == 1 else None
        if hit:
            named.add(os.path.normpath(hit))
        else:
            missing.add(os.path.relpath(os.path.join(d, m), C))
open(os.path.join(C, "temp", "head_named.txt"), "w").write("\n".join(sorted(named)) + "\n")

res = dict(copy=REPO, files=len(pyfiles), unresolved=unresolved, via_repo_dirs=via_repo, parse_errors=parse_err, readme_named=len(named), readme_missing=sorted(missing))
json.dump(res, open(os.path.join(C, "temp", "head_imports.json"), "w"), indent=1)
hard = [u for u in unresolved if u["live"] and not u["guarded"]]
print(f"head_imports: {len(pyfiles)} .py files; imports unresolved in the copy: {len(unresolved)} (in the LIVE tree, unguarded: {len(hard)}; guarded: {sum(u['guarded'] for u in unresolved)}); "
      f"resolved only through another folder of the copy (a sys.path insert): {len(via_repo)}; parse errors {len(parse_err)}; README-named modules {len(named)}, named but not found {len(missing)}")
for u in hard:
    print(f"  NOT IN HEAD, IN THE LIVE TREE: {u['file']}:{u['line']} imports {u['name']} -> live {u['live'][:3]}")
for u in [u for u in unresolved if not u["live"] and not u["guarded"]]:
    print(f"  unresolved anywhere (dead or optional): {u['file']}:{u['line']} {u['name']}")
for m in sorted(missing):
    print(f"  README NAMES A FILE NOT FOUND IN HEAD: {m}")
