"""scratch (2026-11-15, qualitative report): candidate lists per category from the per-frame table (counts of the cards) and the maneuver counts (pass 1); prints, writes nothing."""
import json
import os
import pickle
import statistics as S

import numpy as np

R = os.environ["BFLC_ROOT"]
H = f"{R}/temp/qual_report_1115"
rows = json.load(open(f"{H}/perframe_table.json"))
P1 = pickle.load(open(f"{H}/pass1.pkl", "rb"))
FR, SC = P1["frames"], P1["scenes"]
print("count mismatches vs the stored cards:", len(P1["mismatch"]))
for r, f in zip(rows, FR):
    assert r["token"] == f["token"]
    for s in ("B", "E312", "E324s"):
        r[s + "_mok"], r[s + "_mn"] = f["sys"][s]["man_ok"], f["sys"][s]["man_n"]
print("scenes:")
for sid, v in SC.items():
    print(f"  {sid:2d} {v['name']} n={v['n']:3d}  {v['description']}")
nan = lambda x: x != x  # noqa: E731


def show(r, extra=""):
    print(f"k{r['k']:4d} row{r['row']:5d} sc{r['scene']:2d} {r['token'][:8]} | topo B {r['B_topo']:.2f} E312 {r['E312_topo']:.2f} E324s {r['E324s_topo']:.2f} | ownR B {r['B_ownR']:.2f} 312 {r['E312_ownR']:.2f} 324 {r['E324s_ownR']:.2f} ownP {r['E324s_ownP']:.2f} ownLen {r['ownlen']:.0f} | far {r['B_far']:.2f}/{r['E324s_far']:.2f} | dup {r['B_dup']:.2f}/{r['E324s_dup']:.2f} | lf {r['E324s_lf']}/{r['E324s_lt']} | ngt {r['E324s_ngt']:.0f} | man {r['B_mok']}/{r['B_mn']} {r['E312_mok']}/{r['E312_mn']} {r['E324s_mok']}/{r['E324s_mn']} {extra}")


t = np.array([r["E324s_topo"] for r in rows])
print("E324s per-frame topo F1: mean %.3f median %.3f p25 %.3f p75 %.3f p90 %.3f p10 %.3f" % (t.mean(), np.median(t), *np.percentile(t, [25, 75, 90, 10])))
print("frames: topo F1 < 0.5:", int((t < 0.5).sum()), " >= 0.8:", int((t >= 0.8).sum()))
print("ownR E324s frame-mean %.3f, frames with ownR < 0.8: %d, < 0.5: %d, no tier-1: %d" % (np.nanmean([r["E324s_ownR"] for r in rows]), sum(1 for r in rows if r["E324s_ownR"] < 0.8), sum(1 for r in rows if r["E324s_ownR"] < 0.5), sum(1 for r in rows if nan(r["E324s_ownR"]))))

print("\n== (1) typical good: E324s topo near p75, ownR >= .95, ownP >= .95, ngt >= median")
p75 = np.percentile(t, 75)
ngtm = np.median([r["E324s_ngt"] for r in rows])
c = [r for r in rows if r["E324s_ownR"] >= 0.95 and r["E324s_ownP"] >= 0.95 and r["E324s_ngt"] >= ngtm]
for r in sorted(c, key=lambda r: abs(r["E324s_topo"] - p75))[:8]:
    show(r)

print("\n== (2) junction wins E324s - B (n GT junction >= 3 or maneuvers >= 3), largest gain")
c = [r for r in rows if r["E324s_mn"] >= 3]
for r in sorted(c, key=lambda r: -(r["E324s_topo"] - r["B_topo"]))[:8]:
    show(r, f"gain {r['E324s_topo'] - r['B_topo']:+.2f}")
print("\n   E312 - B largest")
for r in sorted(c, key=lambda r: -(r["E312_topo"] - r["B_topo"]))[:5]:
    show(r, f"gain312 {r['E312_topo'] - r['B_topo']:+.2f}")

print("\n== (3) far lanes missed: E324s far R@1 lowest among frames with far GT >= 50 pts, own lane noted")
c = [r for r in rows if not nan(r["E324s_far"])]
for r in sorted(c, key=lambda r: r["E324s_far"])[:12]:
    show(r)

print("\n== (4) maneuver missing: E324s realised share lowest, maneuvers >= 4")
c = [r for r in rows if r["E324s_mn"] >= 4]
for r in sorted(c, key=lambda r: (r["E324s_mok"] / r["E324s_mn"], -r["E324s_mn"]))[:10]:
    show(r)
print("   maneuvers per frame: frames with >=1:", sum(1 for r in rows if r["E324s_mn"] >= 1), " total realised E324s %d / %d, E312 %d, B %d" % (sum(r["E324s_mok"] for r in rows), sum(r["E324s_mn"] for r in rows), sum(r["E312_mok"] for r in rows), sum(r["B_mok"] for r in rows)))

print("\n== (5) duplicates / false lanes: E324s dup share highest (dupn >= some), and lowest precision")
c = [r for r in rows if r["E324s_predlen"] > 100]
for r in sorted(c, key=lambda r: -(r["E324s_dup"] if not nan(r["E324s_dup"]) else 0))[:8]:
    show(r)
print("   topo P lowest")
for r in sorted(rows, key=lambda r: r["E324s_topo"])[:8]:
    show(r)

print("\n== (7) own lane wrong: ownR lowest with ownLen >= 20 m; own lane right: ownR=1 & ownP=1 & ownLen >= 40")
c = [r for r in rows if not nan(r["E324s_ownR"]) and r["ownlen"] >= 20]
for r in sorted(c, key=lambda r: r["E324s_ownR"])[:12]:
    show(r)
