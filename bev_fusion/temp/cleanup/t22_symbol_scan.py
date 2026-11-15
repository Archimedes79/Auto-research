"""scratch (Image Encoder, clean-base round 2026-11-10): every def / class / method / module-level name of my files and where else its name appears (token match, all text files of the repository).
Each file is tokenised once, so a lookup is a dict access.
usage: python3 bev_fusion/temp/cleanup/t22_symbol_scan.py FILE [FILE ...]   (from the repository root; plain python3)"""
import ast
import collections
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXT = {".py", ".sh", ".md", ".json", ".csv", ".txt", ".ps1"}
SKIP = {".git", "__pycache__", "runs", "data", "wsl", "node_modules", "_to_delete", "results", "temp_cache"}
TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
index = collections.defaultdict(dict)                         # token -> {relpath: count}
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in SKIP]
    for f in fn:
        if os.path.splitext(f)[1] not in EXT:
            continue
        p = os.path.join(dp, f)
        try:
            if os.path.getsize(p) > 2_000_000:
                continue
            t = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        for tok, n in collections.Counter(TOK.findall(t)).items():
            index[tok][rel] = n


def kind(rel):
    base = rel.split("/")[-1]
    if "/tests/" in rel or base.startswith("test_") or "selftest" in base or base.startswith("bev_selftest"):
        return "test"
    if rel.endswith((".md", ".csv", ".json", ".txt")):
        return "text"
    if "/temp/" in rel or rel.startswith("temp/"):
        return "scratch"
    return "code"


for f in sys.argv[1:]:
    own = os.path.relpath(os.path.abspath(f), ROOT).replace("\\", "/")
    tree = ast.parse(open(os.path.abspath(f), encoding="utf-8").read())
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.append((node.name, node.lineno))
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and not (sub.name.startswith("__") and sub.name.endswith("__")):
                        names.append((node.name + "." + sub.name, sub.lineno))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.append((t.id, node.lineno))
    print("=====", own, len(names), "names")
    for full, ln in names:
        short = full.split(".")[-1]
        u = dict(index.get(short, {}))
        selfn = u.pop(own, 0)
        ks = {}
        for rel, n in u.items():
            ks.setdefault(kind(rel), []).append(rel.split("/")[-1])
        tag = ",".join(f"{k}:{len(v)}" for k, v in sorted(ks.items())) or "-"
        flag = ""
        if not ks.get("code") and not ks.get("text"):
            flag = "  <== " + ("own file only" if selfn > 1 else "NO USE / def only") + (" + tests" if ks.get("test") else "") + (" + scratch" if ks.get("scratch") else "")
        elif not ks.get("code"):
            flag = "  <== text/docs only" + (" + tests" if ks.get("test") else "")
        print(f"  {ln:4d} {full:40s} self {selfn:2d} | {tag}{flag}")
