"""survey of the tool calls of a session transcript that touched C:\\ML-Experiment (recovery of files removed by a rollback)"""
import collections
import json
import os
import sys

p = sys.argv[1]
names = collections.Counter()
paths = collections.OrderedDict()
n = 0
with open(p, encoding="utf-8", errors="replace") as f:
    for line in f:
        n += 1
        try:
            d = json.loads(line)
        except Exception:
            continue
        m = d.get("message")
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for b in c:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                names[b.get("name")] += 1
                inp = b.get("input", {})
                fp = inp.get("file_path") or inp.get("path")
                if fp and "ML-Experiment" in str(fp):
                    paths.setdefault(fp, collections.Counter())[b.get("name")] += 1
print("lines", n)
print(dict(names))
missing, present = [], []
for fp, cnt in paths.items():
    w = os.path.exists(fp.replace("C:\\", "/mnt/c/").replace("\\", "/")) if fp.startswith("C:") else os.path.exists(fp)
    (present if w else missing).append((fp, dict(cnt)))
print("files written/edited with the Write/Edit tools:", len(paths), "missing now:", len(missing), "present:", len(present))
print("--- MISSING")
for fp, c in missing:
    print(fp, c)
