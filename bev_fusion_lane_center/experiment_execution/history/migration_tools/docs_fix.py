"""Fix references to the old scripts/ folder in the project documents (md), using where.json (file name -> new folder).   Run: _run.sh docs_fix.py [--go]"""
import json
import os
import re
import sys

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
where = json.load(open("/mnt/c/ML-Experiment/restructure/where.json"))
where.update({"e2e_catalogue.py": "tools/catalogue", "e2e_basis.py": "tools/catalogue", "build_experiment_table.py": "tools/catalogue", "p16_common.py": "training/data", "p16_final_eval.py": "evaluation/drivers"})
GO = "--go" in sys.argv
FILES = ["README.md", "CONVENTIONS.md", "experiment_execution/README.md", "experiment_execution/PROGRAM.md", "experiment_execution/PRACTICE_NOTES.md", "evaluation/README.md", "evaluation/PROTOCOL.md",
         "evaluation/TEST_USAGE_LOG.md", "evaluation/lane_grid/README.md", "evaluation/sparse_lanes/README.md", "data/README.md", "models/README.md", "runs/README.md", "literature/README.md"]
FILES += [f"experiments/{d}/README.md" for d in os.listdir(f"{R}/experiments") if os.path.isdir(f"{R}/experiments/{d}")]
n = 0
for f in FILES:
    p = f"{R}/{f}"
    if not os.path.exists(p):
        continue
    s = open(p, encoding="utf-8").read()
    o = s
    s = re.sub(r"scripts/_py14\.sh \.\./evaluation/(lane_grid|sparse_lanes)/selftest\.py", r"tools/env/_py14.sh \1/selftest.py", s)
    s = s.replace("bev_fusion_lane_center/scripts/_py14.sh", "bev_fusion_lane_center/tools/env/_py14.sh")
    s = re.sub(r"scripts/_py14\.sh \.\./tools/(viz|experiment)/selftest\.py", r"tools/env/_py14.sh \1/selftest.py", s)
    s = s.replace("scripts/_py14.sh", "tools/env/_py14.sh").replace("scripts/_pyr.sh", "tools/env/_pyr.sh")
    s = s.replace("scripts/data_prep/", "training/data/prep/")
    s = s.replace("../../scripts", "../../training")

    def sub(m):
        name = m.group(1)
        d = where.get(name)
        return f"{d}/{name}" if d else m.group(0)

    s = re.sub(r"scripts/([A-Za-z0-9_]+\.(?:py|sh|ps1))", sub, s)
    if s != o:
        n += 1
        if GO:
            open(p, "w", encoding="utf-8").write(s)
print(("fixed " if GO else "would fix "), n, "files")
