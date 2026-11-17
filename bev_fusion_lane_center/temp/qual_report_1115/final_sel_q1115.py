"""scratch (2026-11-15): the written selection rules -> spec.json for render_q1115.py (frames chosen by per-frame numbers of the stored counts, not by eye)."""
import json, os, pickle
import numpy as np
R = os.environ["BFLC_ROOT"]; H = f"{R}/temp/qual_report_1115"
rows = json.load(open(f"{H}/perframe_table.json"))
P1 = pickle.load(open(f"{H}/pass1.pkl", "rb"))
c324 = pickle.load(open(f"{R}/runs/e324_s_s0/ends_card_snap_thr0.1.counts.pkl", "rb"))
c312 = pickle.load(open(f"{R}/runs/e312_p_s0/ends_card_snap_thr0.1.counts.pkl", "rb"))
nan = lambda x: x != x
for r, f in zip(rows, P1["frames"]):
    for s in ("B", "E312", "E324s"):
        r[s + "_mok"], r[s + "_mn"] = f["sys"][s]["man_ok"], f["sys"][s]["man_n"]
    k = r["k"]
    r["far_m"] = c324["head"][k]["topo"]["rng"][2][1] / 2          # far GT in metres (points every 0.5 m)
    r["falseH"] = c324["head"][k]["rec"]["rs"]["false_len"]; r["predH"] = c324["head"][k]["rec"]["rs"]["pred_len"]
    r["dupH_len"] = c324["head"][k]["topo"]["dup"][0] / 2; r["dupE_len"] = c312["head"][k]["topo"]["dup"][0] / 2
t = np.array([r["E324s_topo"] for r in rows]); p75 = np.percentile(t, 75); ngtm = np.median([r["E324s_ngt"] for r in rows])
pick = {}
c = [r for r in rows if r["E324s_ownR"] >= .95 and r["E324s_ownP"] >= .95 and r["E324s_ngt"] >= ngtm]
pick["01_typical_good_own_lane_right"] = min(c, key=lambda r: abs(r["E324s_topo"] - p75))
c = [r for r in rows if r["E324s_mn"] >= 3]
pick["02_junction_win_E312_scene41_row1631"] = max(c, key=lambda r: r["E312_topo"] - r["B_topo"])
pick["03_junction_win_E324s_largest_gain"] = max(c, key=lambda r: r["E324s_topo"] - r["B_topo"])
c = [r for r in rows if r["far_m"] >= 50 and not nan(r["E324s_far"])]
pick["04_far_lanes_missed"] = min(c, key=lambda r: r["E324s_far"])
c = [r for r in rows if r["E324s_mn"] >= 4]
pick["05_junction_maneuvers_missing"] = min(c, key=lambda r: (r["E324s_mok"] / r["E324s_mn"], -r["E324s_mn"]))
pick["06_false_lanes_largest_false_length"] = max(rows, key=lambda r: r["falseH"])
c = [r for r in rows if r["E324s_dup"] >= .45 and r["E324s_dup"] - r["E312_dup"] >= .15]
print("duplicate candidates:", [(r["k"], round(r["dupH_len"])) for r in c])
pick["07_duplicates_E324s_adds"] = max(c, key=lambda r: r["dupH_len"])
c = [r for r in rows if not nan(r["E324s_ownR"]) and r["ownlen"] >= 20]
pick["08_own_lane_worst_frame"] = min(c, key=lambda r: r["E324s_ownR"])
# the scene with the lowest pooled own-lane recall of E324 s, its worst frame
sc = {}
for r in rows:
    if r["ownlen"] > 0:
        a = c324["head"][r["k"]]["tier"]["rec"][0]; sc.setdefault(r["scene"], [0, 0]); sc[r["scene"]][0] += a[0]; sc[r["scene"]][1] += a[1]
w = min(sc, key=lambda s: sc[s][0] / sc[s][1])
print("weakest own-lane scene:", w, sc[w][0] / sc[w][1])
pick["09_own_lane_weakest_scene_worst_frame"] = min([r for r in rows if r["scene"] == w and r["ownlen"] >= 20], key=lambda r: r["E324s_ownR"])
spec = []
for name, r in pick.items():
    print(f"{name}: k{r['k']} row{r['row']} scene{r['scene']} {r['token'][:8]}  topo {r['B_topo']:.2f}/{r['E312_topo']:.2f}/{r['E324s_topo']:.2f}")
    spec.append(dict(name=name, k=r["k"]))
json.dump(spec, open(f"{H}/spec_final.json", "w"), indent=1)
print("p75", p75, "median", np.median(t))
