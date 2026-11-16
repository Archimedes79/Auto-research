import glob
import json
import os

for p in sorted(glob.glob("experiment_execution/runs/E352.s0*/run.json")):
    d = json.load(open(p))
    keep = {k: d[k] for k in d if k in ("id", "status", "started", "ended", "start", "end", "elapsed", "elapsed_s", "exit_code", "wall_s", "seconds", "t_start", "t_end")}
    print(os.path.basename(os.path.dirname(p)), keep or sorted(d.keys())[:30])
print(os.cpu_count())
