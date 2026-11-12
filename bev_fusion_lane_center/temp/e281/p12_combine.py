"""E281 P12: the outcome of the pre-check from the two result files (temp/e281/p12_s0.json, p12_s1.json), by the rules of card section 3 (written before the run):
    GO         the gain AUC(S2) - AUC(S1) is at least 0.05 in both folds on both seeds, pooled OR in the decision band
    STOP-HAND  GO fails but some variant gains at least +0.10 in long_persist on both seeds within the floors (topo F1 0.01, F1@1.0 0.01, junction F1 0.02, lanes found 0.015, pair-rule long precision 0.02)
    STOP       otherwise
Arm h = the variant with the largest mean persistence gain among those within the floors on BOTH seeds (else V1, 'with loss').
    wsl bash tools/env/_py14.sh temp/e281/p12_combine.py"""
import json
import os

ROOT = os.environ["BFLC_ROOT"]
R = {s: json.load(open(os.path.join(ROOT, "temp", "e281", f"p12_s{s}.json"))) for s in (0, 1)}
FLOORS = {"topo_F1": 0.01, "F1@1.0": 0.01, "junction_F1": 0.02, "lanes_found": 0.015, "long_precision_pair": 0.02}
out = {}
# test 1
g = {s: R[s]["B"]["card_label"]["gain S1 -> S2"] for s in (0, 1)}
pooled = all(x >= 0.05 for s in (0, 1) for x in g[s]["pooled"])
band = all(x >= 0.05 for s in (0, 1) for x in g[s]["band"])
out["test1"] = dict(gain_pooled={s: g[s]["pooled"] for s in (0, 1)}, gain_band={s: g[s]["band"] for s in (0, 1)}, pooled_ok=pooled, band_ok=band, GO=bool(pooled or band))
# test 2
variants = [v for v in R[0]["C"] if v != "none"]
rows = {}
for v in variants:
    d = {}
    ok = True
    for s in (0, 1):
        n, c = R[s]["C"]["none"], R[s]["C"][v]
        d[s] = dict(dP=c["long_persist"] - n["long_persist"], **{k: c[k] - n[k] for k in FLOORS})
        ok &= all(d[s][k] >= -FLOORS[k] for k in FLOORS)
    rows[v] = dict(delta=d, within_floors=bool(ok), mean_dP=(d[0]["dP"] + d[1]["dP"]) / 2, both_ge_0_10=bool(d[0]["dP"] >= 0.10 and d[1]["dP"] >= 0.10))
out["test2"] = rows
cand = [v for v in variants if rows[v]["within_floors"]]
h = max(cand, key=lambda v: rows[v]["mean_dP"]) if cand else variants[0]
out["h"] = dict(variant=h, with_loss=not cand)
hand = any(rows[v]["within_floors"] and rows[v]["both_ge_0_10"] for v in variants)
out["outcome"] = "GO" if out["test1"]["GO"] else ("STOP-HAND" if hand else "STOP")
json.dump(out, open(os.path.join(ROOT, "temp", "e281", "p12_outcome.json"), "w"), indent=1)
print("test 1 gains (S2 - S1), folds 0/1, pooled:", {s: [round(x, 3) for x in g[s]["pooled"]] for s in (0, 1)}, " band:", {s: [round(x, 3) for x in g[s]["band"]] for s in (0, 1)})
print("%-30s %9s %9s %s" % ("variant", "dP s0", "dP s1", "worst loss vs floors (s0 / s1) -> within floors"))
for v in variants:
    r = rows[v]
    w = {s: min(r["delta"][s][k] + FLOORS[k] for k in FLOORS) for s in (0, 1)}
    print("%-30s %+9.3f %+9.3f   %+.3f / %+.3f  -> %s" % (v, r["delta"][0]["dP"], r["delta"][1]["dP"], w[0], w[1], r["within_floors"]))
print("arm h:", out["h"], " OUTCOME:", out["outcome"])
