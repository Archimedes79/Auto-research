import sys, re
sys.argv = [sys.argv[0], "999"]
exec(open("/mnt/c/Auto-research/bev_fusion/temp/cleanup/audit_a/symbol_scan.py").read().split("maxo = int")[0])
import os
for path in scope_files:
    if path not in text:
        continue
    lines = text[path].split("\n")
    for name, l0, l1, qual, kind in symbols(text[path]):
        if name in SKIP or (name.startswith("__") and name.endswith("__")):
            continue
        n_own = counts[path][name]
        seg = Counter(tok.findall("\n".join(lines[l0 - 1:l1])))
        own_other = n_own - seg[name]
        oc, ts, tx = [], [], []
        for f, c in counts.items():
            if f == path or not c[name]:
                continue
            k = kind_of(f)
            (ts if k == "test" else tx if k == "text" else oc).append(f)
        # code users outside the scope files themselves and outside this file
        oc_ext = [f for f in oc if f not in scope_files]
        if len(oc_ext) == 0:
            print(f"{os.path.basename(path)}:{l0}-{l1} {qual} [{kind}] own={own_other} scope_users={[os.path.basename(f) for f in oc]} tests={len(ts)}:{[os.path.basename(f) for f in ts][:3]} text={len(tx)}")
