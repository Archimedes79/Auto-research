"""scratch: top-level functions / classes / methods of training/*.py (not legacy) that no other .py file of the repo mentions and that the own file
mentions only inside _selftest / _selftest_* (or not at all). Names only; a lead decides."""
import ast
import glob
import os
import re
import sys

root = os.path.abspath(sys.argv[1])                               # worktree root
files = [p for p in glob.glob(os.path.join(root, "**", "*.py"), recursive=True) if "/temp/" not in p.replace("\\", "/") and "_to_delete" not in p]
text = {p: open(p, encoding="utf-8", errors="replace").read() for p in files}
mods = sorted(glob.glob(os.path.join(root, "bev_fusion", "training", "*.py")))
for path in mods:
    src = text.get(path) or open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    selftest_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and (node.name.startswith("_selftest") or node.name in ("main", "_cli")):
            selftest_lines.update(range(node.lineno, node.end_lineno + 1))
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_selftest"):
            names.append((node.name, node.lineno))
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and not sub.name.startswith("__") and sub.name != "forward":
                        names.append((f"{node.name}.{sub.name}", sub.lineno, sub.name))
    for item in names:
        name, line = item[0], item[1]
        short = item[2] if len(item) > 2 else name
        pat = re.compile(r"\b" + re.escape(short) + r"\b")
        other = 0
        for p, t in text.items():
            if p == path:
                continue
            other += len(pat.findall(t))
        own_live = 0
        for i, ln in enumerate(src.splitlines(), 1):
            if i == line or i in selftest_lines:
                continue
            if pat.search(ln):
                own_live += 1
        if other == 0 and own_live == 0:
            print(f"{os.path.relpath(path, root)}:{line}: {name}")
