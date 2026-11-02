"""Progress of t10_dual_long_s0: step, recent speed (rows of the second attempt only), end estimate, grad-norm spikes. Run from bev_fusion in WSL: python3 temp/t10/progress.py"""
import json
import statistics as st
import time

RESUME = time.mktime(time.strptime("2026-11-01 12:11:48", "%Y-%m-%d %H:%M:%S"))
rows = [json.loads(line) for line in open("runs/t10_dual_long_s0/log.jsonl")]
tr = [r for r in rows if "val" not in r and "t" in r and "grad_norm" in r]
new = [r for r in tr if r["t"] > RESUME]
a, b = new[-11], new[-1]
sps = (b["t"] - a["t"]) / (b["step"] - a["step"])
rem = 6000 - b["step"]
print("now", time.strftime("%H:%M"), "| step", b["step"], "| recent s/step %.2f" % sps, "| end about", time.strftime("%H:%M", time.localtime(b["t"] + rem * sps)), "| total %.2f" % b["total"])
g = [(r["step"], r["grad_norm"]) for r in new[-60:]]
med = st.median(v for _, v in g)
print("grad median %.1f" % med, "| spikes > 5x:", [(s, round(v)) for s, v in g if v > 5 * med])
