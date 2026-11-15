"""primary-source numbers for section 7 of plan/head_design.md (the critic's facts findings 9 - 17), read from the stored files"""
import json
import os
import statistics as S

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT)

print("== response probes (relative change of the stage output when an input is removed)")
for run in ("t16_small_s0", "t20_base_s0"):
    p = f"runs/{run}/eval/modules/response.json"
    if not os.path.exists(p):
        print(run, "no response.json")
        continue
    d = json.load(open(p))
    cfg = json.load(open(f"runs/{run}/config.json"))
    st = None
    sp = f"runs/{run}/eval/modules/static_check.json"
    if os.path.exists(sp):
        st = json.load(open(sp))
    print(run, "steps", cfg["steps"], {k: {a: round(b, 3) for a, b in v.items() if a.startswith("OUT") or a == "fusion:out"} for k, v in d.items()})

print("== Kendall s at the end of the runs")
for run in ("t10_dual_s0", "t10_dual_s1", "t11_vel_hurdle_acc2", "t16_full_s0", "t17_data", "t18_fix", "t20_base_s0"):
    p = f"runs/{run}/log.jsonl"
    if not os.path.exists(p):
        print(run, "no log")
        continue
    rows = [json.loads(line) for line in open(p)]
    tr = [r for r in rows if "s.areas" in r]
    r = tr[-1]
    s = [r["s." + g] for g in ("areas", "lines", "objects", "motion")]
    import math
    print(f"{run:22s} step {r['step']:5d} s = {tuple(round(x, 3) for x in s)}  weights exp(-s) = {tuple(round(math.exp(-x), 2) for x in s)}")

print("== t20 gradient / update ratios, medians of steps 5000-6000")
rows = [json.loads(line) for line in open("runs/t20_base_s0/log.jsonl")]
pr = [r for r in rows if "grad.radar" in r and 5000 <= r["step"] <= 6000]
mods = sorted({k[5:] for r in pr for k in r if k.startswith("grad.")})
g = {m: S.median(r["grad." + m] for r in pr if "grad." + m in r) for m in mods}
u = {m: S.median(r["upd." + m] for r in pr if "upd." + m in r) for m in mods}
for m in sorted(mods, key=lambda m: -g[m]):
    print(f"  {m:14s} grad {g[m]:8.3f} (radar/this {g['radar'] / g[m]:5.2f})   upd/w {u[m]:.2e} (radar/this {u['radar'] / u[m]:5.2f})")
early = [r for r in rows if "grad.radar" in r and 500 <= r["step"] <= 1500]
late = pr
print("  upd/w early (500-1500) vs late for radar:", "%.1e %.1e" % (S.median(r["upd.radar"] for r in early), S.median(r["upd.radar"] for r in late)), " fall x%.0f" % (S.median(r["upd.radar"] for r in early) / S.median(r["upd.radar"] for r in late)))

print("== velocity L1 share of the object group (t20: objhead.vel@+0.0 / group.objects)")
tr = [r for r in rows if "objhead.vel@+0.0" in r and "group.objects" in r]
n = len(tr) // 5
print("  first 20 %% of steps %.2f, last 20 %% %.2f" % (S.median(r["objhead.vel@+0.0"] / r["group.objects"] for r in tr[:n]), S.median(r["objhead.vel@+0.0"] / r["group.objects"] for r in tr[-n:])))
print("== reconstruction weight at the end of 6000 steps: floor + (1 - floor) * 0.5 ** (1 / 0.2) =", round(0.05 + 0.95 * 0.5 ** (1 / 0.2), 3))
