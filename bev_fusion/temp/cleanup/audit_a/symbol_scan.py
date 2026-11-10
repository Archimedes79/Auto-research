"""audit scratch (fast): for every def / class / method / module-level assignment in the scope files, count identifier references per file
(token Counter), split into: own file outside the definition span, other training code, tests, other code, text.
usage: symbol_scan.py [max_own_other]  -> prints symbols with NO reference in other code files and own_other <= max."""
import ast
import os
import re
import subprocess
import sys
from collections import Counter

REPO = "/mnt/c/Auto-research"
SCOPE = "model losses loss_extras layers interface fusion bev_trunk blocks_new fallbacks radar_bev radar_encoder splat temporal temporal_fusion motion_block object_head timed_head extract lane_branch lift_pd bevfusion_camera camera_encoder launch_saving".split()
files = subprocess.run(["git", "-C", REPO, "ls-files"], capture_output=True, text=True).stdout.split("\n")
files = [f for f in files if f and f.endswith((".py", ".md", ".json", ".csv", ".txt", ".sh", ".yaml", ".yml", ".ps1")) and not f.startswith(("bev_fusion/temp/", "temp/"))]
tok = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
text, counts = {}, {}
for f in files:
    p = os.path.join(REPO, f)
    try:
        if os.path.getsize(p) > 3_000_000:
            continue
        s = open(p, encoding="utf-8", errors="replace").read()
    except Exception:
        continue
    text[f] = s
    counts[f] = Counter(tok.findall(s))
legacy = [f for f in files if f.startswith("bev_fusion/training/legacy/") and f.endswith(".py")]
scope_files = [f"bev_fusion/training/{m}.py" for m in SCOPE] + legacy
SKIP = {"__init__", "forward", "extra_repr", "reset_parameters"}


def kind_of(f):
    b = os.path.basename(f)
    if "/tests/" in f or "selftest" in b or b.startswith("test_"):
        return "test"
    return "code" if f.endswith(".py") else "text"


def symbols(src):
    tree = ast.parse(src)
    out = []

    def visit(node, prefix, depth):
        for n in ast.iter_child_nodes(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.append((n.name, n.lineno, n.end_lineno, prefix + n.name, type(n).__name__))
                if isinstance(n, ast.ClassDef):
                    visit(n, prefix + n.name + ".", depth + 1)
            elif depth == 0 and isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        out.append((t.id, n.lineno, n.end_lineno, t.id, "Assign"))
    visit(tree, "", 0)
    return out


maxo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
mode = sys.argv[2] if len(sys.argv) > 2 else "nocode"
for path in scope_files:
    if path not in text:
        continue
    lines = text[path].split("\n")
    for name, l0, l1, qual, kind in symbols(text[path]):
        if name in SKIP or (name.startswith("__") and name.endswith("__")):
            continue
        n_own = counts[path][name]
        inside = len(tok.findall("\n".join(lines[l0 - 1:l1]))) and Counter(tok.findall("\n".join(lines[l0 - 1:l1])))[name]
        own_other = n_own - inside
        oc, ts, tx = [], [], []
        for f, c in counts.items():
            if f == path or not c[name]:
                continue
            k = kind_of(f)
            (ts if k == "test" else tx if k == "text" else oc).append(f)
        if mode == "nocode" and not oc and own_other <= maxo:
            print(f"{os.path.basename(path)}:{l0}-{l1} {qual} [{kind}] own_other={own_other} tests={[os.path.basename(f) for f in ts][:4]}({len(ts)}) text={[os.path.basename(f) for f in tx][:3]}({len(tx)})")
