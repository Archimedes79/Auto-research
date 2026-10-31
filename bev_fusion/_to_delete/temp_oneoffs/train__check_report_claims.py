"""One-off: log values quoted in report.md - final Kendall s per group and the group gradient norms at the BEV feature."""
import json

for run in ("t1_r0", "t1_r3_combined"):
    recs = {}
    for l in open(f"runs/{run}/log.jsonl"):
        r = json.loads(l)
        if "val" not in r:
            recs[r["step"]] = r
    last = recs[max(recs)]
    print(run, "last step", last["step"], "s:", {k[2:]: round(v, 3) for k, v in last.items() if k.startswith("s.")})
    for st in (500, 1000, 2000, 3000, 5000):
        r = recs.get(st)
        if r and "diag" in r:
            print("  step", st, "grad norm at BEV feature", dict(zip(r["diag"]["groups"], [f"{x:.1e}" for x in r["diag"]["norm"]])))
