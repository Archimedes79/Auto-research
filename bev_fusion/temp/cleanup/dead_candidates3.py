"""Scratch v2: dead-code candidates with module-aware name resolution (not a proof; a work list).

A module-level function / class (module M, name N) counts as USED when
  - M itself loads N outside its definition (a Name or Attribute load, a decorator, a default),
  - another .py file imports it (`from M import N`, `from pkg.M import N`, `from . import N` inside the package) or reads `alias.N` where alias is M,
  - N occurs inside a string constant of any .py file or as a token in any non-.py text file (dispatch tables, METHODS tuples, json, docs),
  - some file does `from M import *`.
It is TEST-ONLY when every user is a test / selftest file. A module is UNREFERENCED when no file imports it and no text mentions `M.py` or `-m ...M`.
Methods are not looked at. Output: bev_fusion/temp/cleanup/dead_candidates3.txt
"""
import ast
import collections
import os
import re

FILES = [l.strip() for l in open("bev_fusion/temp/cleanup/files.txt", encoding="utf-8") if l.strip()]
TEXT_EXT = (".py", ".sh", ".md", ".json", ".txt", ".yml", ".yaml", ".csv", ".ps1", ".toml", ".cfg")
SYSTEM = ("autoresearch/", "autogt/", "bev_fusion/training/", "bev_fusion/evaluation/", "bev_fusion/runner/", "bev_fusion/tools/",
          "bev_fusion/tests/", "bev_fusion/studies/", "bev_fusion/splits/", "data/NuScenes/tools/")
TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def is_test(p):
    b = os.path.basename(p)
    return "/tests/" in p or b.startswith("test_") or "selftest" in b


def area(p):
    for a in SYSTEM:
        if p.startswith(a):
            parts = p.split("/")
            return "/".join(parts[:2]) if len(parts) > 2 else a.rstrip("/")
    return None


def mod_of(p):
    return os.path.basename(p)[:-3]


texts, trees = {}, {}
for p in FILES:
    if not p.endswith(TEXT_EXT):
        continue
    try:
        if os.path.getsize(p) > 3_000_000:
            continue
        texts[p] = open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        continue
for p, s in texts.items():
    if p.endswith(".py"):
        try:
            trees[p] = ast.parse(s)
        except SyntaxError:
            pass

# string / non-py token index: name -> files
str_use = collections.defaultdict(set)
doc_use0 = collections.defaultdict(set)
for p, s in texts.items():
    if not p.endswith(".py"):
        tgt = str_use if p.endswith((".json", ".sh", ".ps1", ".toml", ".cfg", ".yml", ".yaml")) else doc_use0
        for t in set(TOK.findall(s)):
            tgt[t].add(p)
doc_use = doc_use0
for p, tree in trees.items():
    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr)                 and isinstance(getattr(n.body[0], "value", None), ast.Constant) and isinstance(n.body[0].value.value, str):
            docs.add(id(n.body[0].value))
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            tgt = str_use if (id(n) not in docs and len(n.value) <= 60 and chr(10) not in n.value) else doc_use
            for t in set(TOK.findall(n.value)):
                tgt[t].add(p)

# imports: (module basename, name) -> files ; module basename -> files that import the module
imp_name = collections.defaultdict(set)
imp_mod = collections.defaultdict(set)
star = set()
attr_use = collections.defaultdict(set)         # (module basename, attr) -> files
for p, tree in trees.items():
    alias_mod = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                m = a.name.split(".")[-1]
                imp_mod[m].add(p)
                alias_mod[(a.asname or a.name).split(".")[0] if not a.asname else a.asname] = m
                if "." in a.name and not a.asname:                      # import pkg.mod -> pkg.mod.attr
                    alias_mod[a.name] = m
        elif isinstance(n, ast.ImportFrom):
            base = (n.module or "").split(".")[-1]
            for a in n.names:
                if a.name == "*":
                    star.add(base)
                    imp_mod[base].add(p)
                    continue
                imp_name[(base, a.name)].add(p)
                imp_mod[a.name].add(p)                                   # `from pkg import mod`
                alias_mod[a.asname or a.name] = a.name
                if base:
                    imp_mod[base].add(p)
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            v = n.value
            if isinstance(v, ast.Name) and v.id in alias_mod:
                attr_use[(alias_mod[v.id], n.attr)].add(p)
            elif isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name):
                key = v.value.id + "." + v.attr
                if key in alias_mod:
                    attr_use[(alias_mod[key], n.attr)].add(p)

unref, unused, testonly, docsonly = (collections.defaultdict(list) for _ in range(4))
for p, tree in trees.items():
    if area(p) is None:
        continue
    m = mod_of(p)
    if m in ("__init__", "__main__", "conftest") or m.startswith("test_"):
        continue
    if not imp_mod.get(m) and not is_test(p) and not re.search(r"\b%s\.py\b|-m\s+[\w.]*\b%s\b" % (re.escape(m), re.escape(m)),
                                                                "\n".join(s for q, s in texts.items() if q != p)):
        unref[area(p)].append(p)
    own_loads = collections.Counter()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            own_loads[n.id] += 1
        elif isinstance(n, ast.Attribute):
            own_loads[n.attr] += 1
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        name = node.name
        if name.startswith("__") or name in ("main", "selftest") or name.startswith(("test_", "_selftest")):
            continue
        if own_loads[name] > 0 or m in star or any(m in star for _ in [0]):
            continue
        users = set(imp_name.get((m, name), ())) | set(attr_use.get((m, name), ())) | {q for q in str_use.get(name, ()) if q != p}
        users.discard(p)
        # a name in a string of a file that also is the definition site does not count (handled by discard)
        if not users:
            dusers = {q for q in doc_use.get(name, ()) if q != p}
            if dusers:
                docsonly[area(p)].append(f"{p}:{node.lineno} {name}  <- text: {', '.join(sorted(os.path.basename(q) for q in dusers)[:3])}")
            else:
                unused[area(p)].append(f"{p}:{node.lineno} {name}")
        elif all(is_test(q) for q in users):
            testonly[area(p)].append(f"{p}:{node.lineno} {name}  <- {', '.join(sorted(os.path.basename(q) for q in users)[:3])}")

out = []
for title, d in (("modules no import and no text mention", unref), ("module-level symbols nobody uses", unused), ("symbols only docs / docstrings / long strings mention", docsonly), ("symbols only tests use", testonly)):
    out.append(f"\n## {title}")
    for a in sorted(d):
        out.append(f"\n### {a} ({len(d[a])})")
        out.extend(f"  {x}" for x in sorted(d[a]))
    print(title, sum(len(v) for v in d.values()), {a: len(v) for a, v in sorted(d.items())})
open("bev_fusion/temp/cleanup/dead_candidates3.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
