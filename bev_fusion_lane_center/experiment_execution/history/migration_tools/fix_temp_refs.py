"""After grouping temp/ by loop: references like .../temp/launchers/_run_x.sh -> .../temp/launchers/loop16/_run_x.sh"""
import os
import re
import sys

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
GO = "--go" in sys.argv
loc = {}
for kind in ("analysis", "launchers"):
    for dp, dn, fn in os.walk(f"{R}/temp/{kind}"):
        for f in fn:
            loc[(kind, f)] = os.path.relpath(dp, R).replace(os.sep, "/")
pat = re.compile(rb"(bev_fusion_lane_center/temp/(analysis|launchers))/([A-Za-z0-9_.\-]+)")
n = 0
for base in ("temp", "training", "evaluation", "tools", "reports/build", "experiments"):
    for dp, dn, fn in os.walk(f"{R}/{base}"):
        dn[:] = [d for d in dn if d not in ("__pycache__", "results", "figures")]
        for f in fn:
            if not f.endswith((".py", ".sh", ".ps1")):
                continue
            p = os.path.join(dp, f)
            b = open(p, "rb").read()

            def sub(m):
                kind, name = m.group(2).decode(), m.group(3).decode()
                d = loc.get((kind, name))
                return (b"bev_fusion_lane_center/" + d.encode() + b"/" + m.group(3)) if d else m.group(0)

            nb = pat.sub(sub, b)
            if nb != b:
                n += 1
                if GO:
                    open(p, "wb").write(nb)
print(("fixed " if GO else "would fix "), n, "files")
