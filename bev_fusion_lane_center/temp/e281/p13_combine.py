"""E281 side check (card section 3b): the plain-threshold curve of p13_s<seed>.json interpolated at the delivered long-link count of the hold variants V3 / V4 of p12_s<seed>.json; gap = hold minus plain.
'The previous frame adds something' only if, for V3 or V4, on BOTH seeds, lanes found OR topo F1 is higher than the plain curve by >= 0.010 and neither is lower by more than 0.005 (rule written before the run).
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh temp/e281/p13_combine.py"""
import json
import os

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
D = os.path.join(ROOT, "temp", "e281")
P13 = {s: json.load(open(os.path.join(D, f"p13_s{s}.json"))) for s in (0, 1)}
P12 = {s: json.load(open(os.path.join(D, f"p12_s{s}.json"))) for s in (0, 1)}
M = ("lanes_found", "topo_F1", "F1@1.0", "junction_F1", "long_precision_pair", "long_persist")
HOLD = ("V3 hold p>=0.25 prev>=0.5", "V4 hold p>=0.1 prev>=0.5")
out = {"curve": {}, "gaps": {}}
for s in (0, 1):
    cur = P13[s]["variants"]
    ts = sorted(cur, key=lambda t: float(t), reverse=True)           # 0.5 first: delivered count rises as t falls
    x = np.array([cur[t]["delivered_long"] for t in ts], float)
    out["curve"][s] = {t: {k: cur[t][k] for k in M + ("delivered_long",)} for t in ts}
    for h in HOLD:
        c = P12[s]["C"][h]
        n = float(c["delivered_long"])
        covered = bool(x.min() <= n <= x.max())
        plain = {k: float(np.interp(n, x, np.array([cur[t][k] for t in ts], float))) for k in M} if covered else None
        gap = {k: c[k] - plain[k] for k in M} if covered else None
        out["gaps"][f"{h} | seed {s}"] = dict(delivered=n, covered=covered, hold={k: c[k] for k in M}, plain=plain, gap=gap)
something = False
for h in HOLD:
    ok = True
    for s in (0, 1):
        g = out["gaps"][f"{h} | seed {s}"]["gap"]
        if g is None:
            ok = False
            continue
        up = g["lanes_found"] >= 0.010 or g["topo_F1"] >= 0.010
        no_loss = g["lanes_found"] >= -0.005 and g["topo_F1"] >= -0.005
        ok &= bool(up and no_loss)
    out[f"adds_something {h}"] = bool(ok)
    something |= ok
out["verdict"] = "the previous frame adds something" if something else "the hold is only a lower threshold"
json.dump(out, open(os.path.join(D, "p13_outcome.json"), "w"), indent=1)
for s in (0, 1):
    print(f"--- seed {s}: plain long threshold (B = 0.5)")
    print("t      delivered  lanes   topo    F1@1.0  junc    prec_pair  persist")
    for t, v in out["curve"][s].items():
        print("%-5s  %8d   %.3f   %.3f   %.3f   %.3f   %.3f      %.3f" % (t, v["delivered_long"], v["lanes_found"], v["topo_F1"], v["F1@1.0"], v["junction_F1"], v["long_precision_pair"], v["long_persist"]))
print()
for k, g in out["gaps"].items():
    if not g["covered"]:
        print(f"{k}: count {g['delivered']:.0f} NOT covered by the curve")
        continue
    print(f"{k}: delivered {g['delivered']:.0f}; hold - plain at the same count: " + ", ".join(f"{m} {g['gap'][m]:+.3f}" for m in M))
print("\nRESULT:", out["verdict"], {h: out[f"adds_something {h}"] for h in HOLD})
