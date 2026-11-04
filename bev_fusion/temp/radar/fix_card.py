import subprocess
p = "experiments/t19_radar_memory/experiment.md"
s = open(p, encoding="utf-8").read()
for old, new in (
    ("cost +3 GFLOPs (+2 % of the chain; 115 k parameters at 64 channels, `training/README.md`), ms per frame\n  +2 to +6.",
     "cost +3 GFLOPs (about +2-3 % of the chain; 115 k parameters at 64 channels, `training/README.md`), ms per frame\n  +2 to +6 (a guess)."),
    ("cost: ms per frame (`bev_suite.py time`, idle GPU) and the radar response probe of\n  `evaluation/` (how much the maps / objects change when all detections are removed).",
     "cost: ms per frame (`bev_suite.py time`, idle GPU) and the module probe of the t16 report (how much the stages change when all\n  detections are removed)."),
    ("(+2 % of the chain)", "(about +2-3 % of the chain)")):
    if old in s:
        s = s.replace(old, new, 1)
    else:
        print("anchor not found:", old[:50])
open(p, "w", encoding="utf-8").write(s)
print("card fixed")
