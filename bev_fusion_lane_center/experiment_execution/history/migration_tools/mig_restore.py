"""Bring back staged run entries that live code names literally: {OUT}/<name> (exact) or {OUT}/<prefix>{...} (prefix), plus loop-prefixed families (p10..p17, timing).
Usage: mig_restore.py [--go]"""
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mig_common import *  # noqa: E402,F401,F403

GO = "--go" in sys.argv
ST = f"{R}/_to_delete/runs"
RN = f"{R}/runs"
pat = re.compile(r"\{(?:\w+\.)?(?:OUT|E17|LADDER|D|C\.OUT|PL\.OUT|DH\.OUT)\}/([A-Za-z0-9_\-\.]+)(\{?)")
pat2 = re.compile(r"out/loop2/([A-Za-z0-9_\-\.]+)(\{?)")
exact, prefix = set(), set()
files = []
for d in (f"{R}/scripts", f"{R}/evaluation", f"{R}/tools/viz"):
    for dp, dn, fn in os.walk(d):
        dn[:] = [x for x in dn if x not in ("_archive", "__pycache__")]
        files += [os.path.join(dp, f) for f in fn if f.endswith((".py", ".sh", ".ps1"))]
for p in files:
    s = open(p, encoding="utf-8", errors="replace").read()
    for rx in (pat, pat2):
        for m in rx.finditer(s):
            (prefix if m.group(2) else exact).add(m.group(1))
names = os.listdir(ST)
want = set()
for n in names:
    base = n.rsplit(".", 1)[0] if n.endswith((".log", ".out", ".done")) else n
    if re.match(r"^p1[0-7]", n) or n in ("timing", "figs") or n.startswith(("refiner_stats", "_subset", "_t_order")):
        if n != "figs":
            want.add(n)
    if n in exact or any(n == e or n.startswith(e + ".") for e in exact):
        want.add(n)
    if any(n.startswith(p_) for p_ in prefix if len(p_) >= 4):
        want.add(n)
big = [(n, 0) for n in want]
tot = 0
for n in sorted(want):
    p = f"{ST}/{n}"
    sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(p) for f in fn) if os.path.isdir(p) else os.path.getsize(p)
    tot += sz
print(f"{len(want)} staged entries named by live code: {tot/1e6:.0f} MB")
print(" ".join(sorted(want))[:3000])
if GO:
    for n in sorted(want):
        shutil.move(f"{ST}/{n}", f"{RN}/{n}")
        with open(f"{RS}/MIGRATION_LOG.csv", "a", encoding="utf-8") as lg:
            lg.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},restore,{ST}/{n},{RN}/{n}\n")
    print("restored", len(want))
