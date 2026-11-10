import glob
import os
import re
from collections import defaultdict

cat = defaultdict(lambda: [0, 0])          # category -> [bytes, files]
runs = defaultdict(int)
for ev in glob.glob("runs/*/eval"):
    run = ev.split("/")[1]
    for root, _, files in os.walk(ev, followlinks=False):
        rel = os.path.relpath(root, ev)
        for f in files:
            p = os.path.join(root, f)
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            runs[run] += sz
            top = rel.split(os.sep)[0] if rel != "." else "(top)"
            if top == "suite":
                parts = rel.split(os.sep)
                top = "suite/" + (parts[1] if len(parts) > 1 else "") + ("/" + re.sub(r"_n\d+", "_nN", parts[2]) if len(parts) > 2 and parts[1] == "quant" else "")
            if f.startswith("scene_records"):
                top = "scene_records*.pkl"
            elif top == "(top)":
                top = "(top) " + os.path.splitext(f)[1]
            top = re.sub(r"_n\d+", "_nN", top)
            cat[top][0] += sz
            cat[top][1] += 1
tot = sum(v[0] for v in cat.values())
print(f"{len(runs)} runs with an eval folder, {tot / 1e9:.2f} GB")
for k, (b, n) in sorted(cat.items(), key=lambda kv: -kv[1][0])[:22]:
    print(f"{b / 1e6:8.0f} MB {n:6d} files  {k}")
print("largest runs:", ", ".join(f"{r} {b / 1e6:.0f}" for r, b in sorted(runs.items(), key=lambda kv: -kv[1])[:6]))
print("results folders:", len(glob.glob("runs/*/eval/results/*")))
