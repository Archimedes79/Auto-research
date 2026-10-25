"""After the layout move: rewrite references to the old scripts folder.  where.json maps file name -> new folder.   Run: _run.sh layout_rewrite.py [--go]"""
import json
import os
import re
import sys

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
RS = "/mnt/c/ML-Experiment/restructure"
GO = "--go" in sys.argv
where = json.load(open(f"{RS}/where.json"))
where["e2e_catalogue.json"] = "tools/catalogue"
for f in ("_py14.sh", "_py.sh", "_pyr.sh"):
    where[f] = "tools/env"
ROOTS = [f"{R}/training", f"{R}/evaluation", f"{R}/tools", f"{R}/reports/build", f"{R}/temp", f"{R}/experiments"]
EXT = (".py", ".sh", ".ps1")
PFX = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
pA = re.compile(rb"(bev_fusion_lane_center[/\\]{1,2})scripts([/\\]{1,2})(?:_archive[/\\]{1,2}[A-Za-z0-9_]+[/\\]{1,2}|data_prep[/\\]{1,2})?([A-Za-z0-9_.\-]+)")
pS = re.compile(rb"(\$\{?[A-Za-z_]+\}?)/([A-Za-z0-9_.\-]+\.(?:py|sh|ps1))")
changed = unknown = 0
unk = set()
for root in ROOTS:
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ("__pycache__", "loop17_recovery", "runs", "results", "figures")]
        for f in fn:
            if not f.endswith(EXT):
                continue
            p = os.path.join(dp, f)
            b = open(p, "rb").read()
            nb = b
            # shell wrappers with S=<scripts dir>: $S/name -> absolute new location
            if b"scripts" in b:
                m = re.search(rb"^([A-Za-z_]+)=(/mnt/c/ML-Experiment/bev_fusion_lane_center/scripts)\s*$", b, re.M)
                if m:
                    var = m.group(1)

                    def subS(mm):
                        if mm.group(1).strip(b"${}") != var:
                            return mm.group(0)
                        name = mm.group(2).decode()
                        d = where.get(name)
                        if d is None:
                            unk.add(name)
                            return mm.group(0)
                        return (PFX + "/" + d + "/" + name).encode()

                    nb = pS.sub(subS, nb)

                def subA(mm):
                    name = mm.group(3).decode()
                    d = where.get(name)
                    if d is None:
                        if "." not in name:  # a directory reference (cd .../scripts)
                            return mm.group(1) + b"temp/launchers" + mm.group(2) + mm.group(3) if False else mm.group(0)
                        unk.add(name)
                        return mm.group(0)
                    sep = mm.group(2)
                    return mm.group(1) + d.replace("/", sep.decode()).encode() + sep + mm.group(3)

                nb = pA.sub(subA, nb)
                # bare references to the folder itself
                nb = re.sub(rb"(bev_fusion_lane_center)([/\\]{1,2})scripts(?=[\"'\s]|$)", rb"\1\2training", nb, flags=re.M)
            if nb != b:
                changed += 1
                if GO:
                    open(p, "wb").write(nb)
print(("rewritten" if GO else "would be rewritten"), changed, "files; names without a new location:", sorted(unk)[:40])
