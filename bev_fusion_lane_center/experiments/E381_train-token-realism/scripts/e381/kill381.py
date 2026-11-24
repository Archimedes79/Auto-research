"""E381 card v2.1: the kill checks of a finished arm-n run from its own log (CPU, scratch), BEFORE any read-out. usage: python kill381.py <seed> -> prints the checks, exit 0 = no kill, 3 = kill.
The checks of the JSON kill: finite losses; learning (the mean of the rows at steps 1900 / 2000 below 0.8 x the row at 200); the noise active and right (the mean over the rows of noise_disp and noise_e_abs within 0.8 - 1.2 x
SP sqrt(pi/2), SE sqrt(2/pi)); the gate (the mean over the rows of gt_outside / gt_links at most 3 x the ref's mean); the twin (the CLEAN first-batch loss = the ref's first loss to 1e-3); the run finished its 12000 steps."""
import json
import math
import os
import re
import sys

R = os.environ.get("BFLC_ROOT", ".")
S = int(sys.argv[1])
SP, SE = {0: (0.184, 0.621), 1: (0.181, 0.601)}[S]
REF_FIRST = {0: 4.5516, 1: 4.3306}[S]
REF_GATE = {0: 0.0085, 1: 0.0098}[S]
rd = os.path.join(R, "runs", f"e381_n_s{S}")
rows = [json.loads(x) for x in open(os.path.join(rd, "log.jsonl"))]
log = open(os.path.join(rd, "train.log"), encoding="utf-8").read()
cfg = json.load(open(os.path.join(rd, "config.json")))["args"]
by = {r["step"]: r for r in rows}
mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
checks = {}
checks["args"] = (cfg["noise_pos"], cfg["noise_e"], cfg["head_seed"], cfg["seed"], cfg["steps"]) == (SP, SE, 2 + S, S, 12000), (cfg["noise_pos"], cfg["noise_e"], cfg["head_seed"], cfg["seed"], cfg["steps"])
checks["finished"] = max(by) == 12000 and os.path.exists(os.path.join(rd, ".done")), max(by)
st = json.load(open(os.path.join(rd, "status.json")))
checks["finite"] = st.get("skipped") == 0 and all(math.isfinite(r["total"]) for r in rows), ("skipped non-finite steps", st.get("skipped"), "rollbacks", st.get("rollbacks"))
lr = mean([by[1900]["total"], by[2000]["total"]]) / by[200]["total"]
checks["learning"] = lr < 0.8, round(lr, 3)
d, e = mean([r["noise_disp"] for r in rows]), mean([r["noise_e_abs"] for r in rows])
ed, ee = SP * math.sqrt(math.pi / 2), SE * math.sqrt(2 / math.pi)
checks["noise_pos"] = 0.8 <= d / ed <= 1.2, (round(d, 4), round(ed, 4))
checks["noise_e"] = 0.8 <= e / ee <= 1.2, (round(e, 4), round(ee, 4))
g = mean([r["gt_outside"] / r["gt_links"] for r in rows])
checks["gate"] = g <= 3 * REF_GATE, (round(g, 4), 3 * REF_GATE)
m = re.findall(r"the CLEAN loss of this batch ([0-9.]+)", log)
checks["twin"] = bool(m) and abs(float(m[0]) - REF_FIRST) <= 1e-3, (m[:1], REF_FIRST)
checks["resumes"] = True, log.count("resumed from")
for k, (ok, v) in checks.items():
    print(f"seed {S} {k:10s} {'ok' if ok else 'KILL'}  {v}")
kill = [k for k, (ok, _) in checks.items() if not ok]
print(f"RESULT seed {S}: {'KILL ' + ','.join(kill) if kill else 'no kill'}")
sys.exit(3 if kill else 0)
