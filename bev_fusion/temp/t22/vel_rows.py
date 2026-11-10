"""scratch: velocity / static EPE rows of the rule (control, twin, stand-in, B) from RUN/eval/metrics.json"""
import json, sys, re
runs = ["runs/t21_pair4_s0", "runs/t22_twin_pair4", "runs/t22_standin_pair4", "runs/t22_B_s0"]
def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, p + k + "/"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[p + k] = v
    return out
M = [flat(json.load(open(r + "/eval/metrics.json"))["metrics"]) for r in runs]
pat = re.compile(sys.argv[1]) if len(sys.argv) > 1 else re.compile(r"^vel/")
keys = sorted(k for k in M[0] if pat.search(k))
print(len(keys), "keys matching", pat.pattern)
lim = int(sys.argv[2]) if len(sys.argv) > 2 else 80
for k in keys[:lim]:
    vals = [m.get(k) for m in M]
    if all(v is not None for v in vals):
        print(f"{k:80s} " + " ".join(f"{v:9.4f}" for v in vals) + f"   B-ctl {vals[3]-vals[0]:+8.4f}")
