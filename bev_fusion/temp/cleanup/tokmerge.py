"""Scratch: a word-level three-way merge for the conflicted text files of a cherry-pick (the line-level merge fails when both sides edited the same long table row).

usage: python3 tokmerge.py OURS BASE THEIRS OUT      -> writes OUT; prints the number of token conflicts (0 = clean)
Every file is cut into tokens (words and whitespace runs, one per line), `git merge-file` merges the token lists, the result is joined again.
"""
import re
import subprocess
import sys
import tempfile
import os

ours, base, theirs, out = sys.argv[1:5]


def tokens(path):
    s = open(path, encoding="utf-8", newline="").read()
    parts = re.findall(r"\s+|[^\s]+", s)
    return [p.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r") + "\n" for p in parts]


def untoken(lines):
    s = "".join(l[:-1] if l.endswith("\n") else l for l in lines)
    return s


tmp = tempfile.mkdtemp()
paths = []
for name, p in (("o", ours), ("b", base), ("t", theirs)):
    q = os.path.join(tmp, name)
    open(q, "w", encoding="utf-8", newline="").write("".join(tokens(p)))
    paths.append(q)
r = subprocess.run(["git", "merge-file", "-p", "--diff3", "-L", "ours", "-L", "base", "-L", "theirs"] + paths, capture_output=True, text=True)
if r.returncode < 0 or (not r.stdout and os.path.getsize(ours) > 0):
    print(f"git merge-file failed: rc={r.returncode} {r.stderr.strip()[:300]}")
    sys.exit(2)
merged = r.stdout.split("\n")
merged = [m + "\n" for m in merged[:-1]] + ([merged[-1]] if merged[-1] else [])
conflicts = sum(1 for l in merged if l.startswith("<<<<<<< "))


def decode(l):
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "r": "\r", "\\": "\\"}.get(m.group(1), "\\" + m.group(1)), l)


text = ""
for l in merged:
    t = l[:-1] if l.endswith("\n") else l
    if t.startswith(("<<<<<<<", "=======", ">>>>>>>", "|||||||")) and len(t) >= 7 and t[:7] in ("<<<<<<<", "=======", ">>>>>>>", "|||||||") and (len(t) == 7 or t[7] == " "):
        text += f"\n{t}\n"
    else:
        text += decode(t)
open(out, "w", encoding="utf-8", newline="").write(text)
print(f"token conflicts: {conflicts}")
