import sys, subprocess
sys.path.insert(0, ".")
import evaluation.bev_timing as t
ps = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
print("GPU holders in the live ps listing now:", [x[:100] for x in t.busy_processes(ps, set())])
print("launchers that the old logic counted:", len([ln for ln in ps.splitlines() if "run.py" in ln and "train.py" in ln or "run_ar.sh" in ln]))
