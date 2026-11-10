"""the two identity hashes of the BEV evaluation computed for a tree (old export, new export, live): python idhash.py <repo root of the tree>
evaluator hash = provenance.tree_hash over NUMBER_PATHS (relative names + bytes, CRLF as LF); prediction adapter hash = sha1 over the raw-byte file hashes of _PREDICTION_FILES.
The two path lists are read out of the tree's own bev_adapter.py / bev_eval.py, so this is what that tree computes."""
import hashlib
import importlib.util
import os
import re
import sys

base = os.path.abspath(sys.argv[1])
ROOT = os.path.join(base, "bev_fusion")


def grab(path, pattern, flags=re.M):
    with open(path, encoding="utf-8", newline="") as f:
        src = f.read().replace("\r\n", "\n")                               # the statement only; the hashes below read the files as bytes
    m = re.search(pattern, src, flags)
    assert m, (path, pattern)
    return m.group(0).split("=", 1)[1].strip()


ns = {"os": os, "ROOT": ROOT}
number_paths = eval(grab(os.path.join(ROOT, "evaluation", "bev_adapter.py"), r"^NUMBER_PATHS = .*\\\n.*"), ns)
pred_files = eval(grab(os.path.join(ROOT, "evaluation", "bev_eval.py"), r"^_PREDICTION_FILES = \(.*?\)(?=\s*#)", re.S | re.M), ns)

spec = importlib.util.spec_from_file_location("provenance", os.path.join(base, "autoresearch", "tools", "provenance.py"))
prov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prov)


def file_hash(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


ev = prov.tree_hash([os.path.abspath(p) for p in number_paths], base)
adapter = hashlib.sha1("".join(file_hash(p) for p in pred_files).encode()).hexdigest()[:12]
print(f"{base}\n  evaluator_hash = {ev}\n  prediction adapter = {adapter}")
for p in pred_files:
    print(f"    {os.path.relpath(os.path.abspath(p), base).replace(chr(92), '/')}  {file_hash(p)}")
