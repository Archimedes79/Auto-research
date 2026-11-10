"""Scratch: candidates for dead code in the system folders (not a proof, a work list for the owners).

reads temp/cleanup/files.txt (git ls-files), builds a token index over every tracked text file and reports
  - modules (.py) of the system folders that no other file mentions by name (import, path, command, document)
  - module-level functions / classes whose name occurs nowhere else (only the definition)
  - the same, but only mentioned in tests / selftests (alive for a test only)
Writes temp/cleanup/dead_candidates.txt and prints a count per area.
"""
import ast
import collections
import os
import re
import sys

ROOT = os.getcwd()
FILES = [l.strip() for l in open("bev_fusion/temp/cleanup/files.txt", encoding="utf-8") if l.strip()]
TEXT_EXT = (".py", ".sh", ".md", ".json", ".txt", ".yml", ".yaml", ".csv", ".ps1", ".toml", ".cfg")
SYSTEM = ("autoresearch/", "autogt/", "bev_fusion/training/", "bev_fusion/evaluation/", "bev_fusion/runner/", "bev_fusion/tools/",
          "bev_fusion/tests/", "bev_fusion/studies/", "bev_fusion/splits/", "data/NuScenes/tools/")
TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
tokens = {}          # path -> Counter
texts = {}
for p in FILES:
    if not p.endswith(TEXT_EXT):
        continue
    try:
        if os.path.getsize(p) > 3_000_000:
            continue
        s = open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        continue
    texts[p] = s
    tokens[p] = collections.Counter(TOK.findall(s))

total = collections.Counter()
for c in tokens.values():
    total.update(c)


def is_test(p):
    b = os.path.basename(p)
    return "/tests/" in p or b.startswith("test_") or b.endswith("_selftest.py") or b.startswith("selftest") or "selftest" in b


def area(p):
    for a in SYSTEM:
        if p.startswith(a):
            parts = p.split("/")
            return "/".join(parts[:2]) if a.count("/") == 1 and p.count("/") > 1 else a.rstrip("/")
    return None


out = []
unref_mod, unused_sym, test_only = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
for p in FILES:
    if not p.endswith(".py") or area(p) is None or p not in texts:
        continue
    base = os.path.basename(p)[:-3]
    if base in ("__init__", "__main__", "conftest") or base.startswith("test_"):
        continue
    # module mentioned by name anywhere else?
    others = sum(c[base] for q, c in tokens.items() if q != p)
    if others == 0 and not is_test(p):
        unref_mod[area(p)].append(p)
    try:
        tree = ast.parse(texts[p])
    except SyntaxError:
        out.append(f"SYNTAX {p}")
        continue
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            n = node.name
            if n.startswith("__") or n in ("main", "selftest") or n.startswith("test_") or n.startswith("_selftest"):
                continue
            # occurrences elsewhere than this file's own definition: total - own count + (own count - 1) uses inside the file
            own = tokens[p][n]
            elsewhere = total[n] - own
            if elsewhere == 0 and own <= 1:
                (test_only if False else unused_sym)[area(p)].append(f"{p}:{node.lineno} {n}")
            elif elsewhere > 0 and own <= 1:
                users = [q for q, c in tokens.items() if q != p and c[n] > 0]
                if users and all(is_test(q) for q in users):
                    test_only[area(p)].append(f"{p}:{node.lineno} {n}  <- {', '.join(sorted(os.path.basename(q) for q in users)[:3])}")

def dump(title, d):
    out.append(f"\n## {title}")
    for a in sorted(d):
        out.append(f"\n### {a} ({len(d[a])})")
        out.extend(f"  {x}" for x in sorted(d[a]))

dump("modules no other file mentions by name", unref_mod)
dump("module-level symbols mentioned nowhere else", unused_sym)
dump("module-level symbols that only tests / selftests mention", test_only)
open("bev_fusion/temp/cleanup/dead_candidates.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
for title, d in (("unreferenced modules", unref_mod), ("unused symbols", unused_sym), ("symbols used only by tests", test_only)):
    print(title, sum(len(v) for v in d.values()), {a: len(v) for a, v in sorted(d.items())})
