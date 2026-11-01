"""E253 analysis: median of the replicate medians per configuration, spread, the decision clause (a - c, band 1.0 ms), reproduction of the stored values.

    wsl python3 temp/loop19/analyse_e243.py        (reads runs/e19/e243/<arm>_r<k>.json; writes runs/e19/e243/summary.json)

Arms: a = j1 192/96 (pruning 0.3), c = f1 288/144 rec 0 (0.3), b = j1eq, d = j1rc, e = rc1 288/144 (0.4), f = f1 with time recursion.
Decision (card): KEEP if a is faster than c by >= 1.0 ms with all replicate pairs on the same side; CUT if a is slower by > 1.0 ms; else TIE.
Voided replicates ({"void": true}) are skipped and counted. Reproduction check: new a and c medians within 1.0 ms of the stored loop-18 values (17.15-18.10 and 17.54-18.23 ms).
"""
import glob
import json
import os
import statistics as st

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs", "e19", "e253")
NAMES = dict(a="j1 192/96 fp0.3", c="f1 288/144 fp0.3 (rec 0)", b="j1eq fp0.4", d="j1rc fp0.4", e="rc1 288/144 fp0.4", f="f1 rec 1 (as trained)")
STORED = dict(a=(17.15, 18.10), c=(17.54, 18.23), f=(24.2, 24.2))
rows, void = {}, 0
for arm in NAMES:
    for k in range(1, 9):
        p = f"{D}/{arm}_r{k}.json"
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        if r.get("void"):
            void += 1
            continue
        rows.setdefault(arm, []).append(r)
out = {}
print(f"{'arm':3} {'configuration':26} {'n':>2} {'median ms':>9} {'min':>6} {'max':>6} {'spread':>6} {'asm ms':>7} {'builder':>7} {'points':>7}")
for arm, rs in rows.items():
    ms = [r["ms_per_frame"] for r in rs]
    out[arm] = dict(n=len(ms), median=st.median(ms), lo=min(ms), hi=max(ms), spread=max(ms) - min(ms), asm=st.median(r["asm_ms"] for r in rs),
                    builder=st.median(r["builder_ms"] for r in rs), points=st.mean(r["points"] for r in rs), replicates=ms)
    o = out[arm]
    print(f"{arm:3} {NAMES[arm]:26} {o['n']:>2} {o['median']:9.2f} {o['lo']:6.2f} {o['hi']:6.2f} {o['spread']:6.2f} {o['asm']:7.2f} {o['builder']:7.2f} {o['points']:7.1f}")
verdict = "n/a"
if "a" in out and "c" in out:
    eff = out["a"]["median"] - out["c"]["median"]
    pairs = [x - y for x, y in zip(out["a"]["replicates"], out["c"]["replicates"])]
    faster = all(p <= -1.0 for p in pairs) and eff <= -1.0
    slower = eff > 1.0
    verdict = "KEEP" if faster else "CUT" if slower else "TIE"
    asm_guard = out["a"]["asm"] - out["c"]["asm"] <= 1.0
    print(f"\na - c = {eff:+.2f} ms (median of replicate medians); pairs {[round(p, 2) for p in pairs]}; assembly guard (a - c <= +1.0 ms): {asm_guard} -> {verdict}")
    for arm, (lo, hi) in STORED.items():
        if arm in out:
            print(f"reproduction {arm}: new {out[arm]['median']:.2f} ms vs stored {lo:.2f}-{hi:.2f} ms -> {'ok' if lo - 1.0 <= out[arm]['median'] <= hi + 1.0 else 'MACHINE STATE DIFFERS'}")
if "b" in out and "d" in out:
    print(f"j1rc - j1eq = {out['d']['median'] - out['b']['median']:+.2f} ms (prediction +2.2)")
if "e" in out and "c" in out:
    print(f"rc1 - f1 = {out['e']['median'] - out['c']['median']:+.2f} ms (prediction +2.5)")
if "f" in out and "c" in out:
    print(f"f1 rec 1 - rec 0 = {out['f']['median'] - out['c']['median']:+.2f} ms")
over = [a for a, o in out.items() if o["spread"] > 1.5]
print(f"configurations with replicate spread > 1.5 ms (band withdrawn if any): {over or 'none'}; voided replicates: {void}")
json.dump(dict(arms=out, verdict=verdict, void=void), open(f"{D}/summary.json", "w"), indent=1)

# session validity (referee of E253): replicate SD instead of the max - min spread, reproduction of the stored values, at least five valid replicates per configuration
sds = {a: (st.stdev(o["replicates"]) if o["n"] > 1 else float("nan")) for a, o in out.items()}
repro = all(STORED[a][0] - 1.0 <= out[a]["median"] <= STORED[a][1] + 1.0 for a in STORED if a in out)
enough = all(o["n"] >= 5 for o in out.values()) and len(out) == 6
calm = sum(1 for s in sds.values() if s <= 0.6) >= 5
valid = repro and enough and calm
print(f"replicate SD per configuration (ms): { {a: round(s, 2) for a, s in sds.items()} }; >= 5 valid replicates in all six: {enough}; SD <= 0.6 ms in >= 5 of 6: {calm}; stored values reproduced within 1.0 ms: {repro}")
print(f"SESSION {'VALID' if valid else 'INVALID (the decision clause is not read)'}")
json.dump(dict(arms=out, verdict=verdict if valid else "INVALID", void=void, sds=sds, repro=repro), open(f"{D}/summary.json", "w"), indent=1)
