import os
import re
from collections import defaultdict

OUT = "/mnt/c/ML-Experiment/out/loop2"
ext = defaultdict(lambda: [0, 0])
small = defaultdict(lambda: [0, 0])
live_tokens = set()
SC = "/mnt/c/ML-Experiment/scripts"
import json

cl = json.load(open("/mnt/c/ML-Experiment/restructure/classify.json"))
live = set(cl["live"])
for m in live:
    try:
        s = open(f"{SC}/{m}.py", encoding="utf-8", errors="replace").read()
    except OSError:
        continue
    live_tokens.update(re.findall(r"\be\d{2,3}[a-z0-9_]*", s))
tops = os.listdir(OUT)
hit = [t for t in tops if any(t.startswith(k) or k.startswith(t) for k in live_tokens if len(k) >= 3)]
print("live tokens:", len(live_tokens), "entries matched:", len(hit))
tot_hit = 0
for dp, dn, fn in os.walk(OUT):
    for f in fn:
        p = os.path.join(dp, f)
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        e = os.path.splitext(f)[1].lower() or "(none)"
        ext[e][0] += 1
        ext[e][1] += sz
        if sz <= 2_000_000:
            small[e][0] += 1
            small[e][1] += sz
print("ext: count, MB, | <=2MB: count, MB")
for e, v in sorted(ext.items(), key=lambda kv: -kv[1][1])[:22]:
    print(f"  {e:10s} {v[0]:6d} {v[1]/1e6:10.0f} | {small[e][0]:6d} {small[e][1]/1e6:8.0f}")
print("pngs:", ext[".png"][0])
sz = lambda p: sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(p) for f in fn) if os.path.isdir(p) else os.path.getsize(p)
tot = sum(sz(f"{OUT}/{t}") for t in hit)
print("MB of matched entries:", round(tot / 1e6))
print("matched sample:", sorted(hit)[:80])
