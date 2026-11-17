"""scratch (2026-11-17): the written selection rules for the qualitative pictures of the token champion -> spec_champion_q1117.json + stats_champion_q1117.json.
Frames are chosen from the per-frame numbers (champion: topo F1 and maneuvers from the graphs of runs/e365_s1_s0 at the E366 decode; plain B and E324 s: the stored per-frame table of 2026-11-15), not by eye.
usage: wsl bash tools/env/_py14.sh select_champion_q1117.py"""
import json, os, pickle
import numpy as np

R = os.environ["BFLC_ROOT"]
H15, H17 = f"{R}/temp/qual_report_1115", f"{R}/temp/qual_report_1117"
rows = {r["k"]: r for r in json.load(open(f"{H15}/perframe_table.json"))}
CH = json.load(open(f"{H17}/perframe_table.json"))["sys"]["champion_S1_s0"]
P = pickle.load(open(f"{H17}/pass1_with_champion.pkl", "rb"))
n = len(P["frames"])
ct = np.array(CH["topo_F1"], float)
bt = np.array([rows[k]["B_topo"] for k in range(n)], float)
et = np.array([rows[k]["E324s_topo"] for k in range(n)], float)
ok = {s: np.array([P["frames"][k]["sys"][s]["man_ok"] for k in range(n)], float) for s in ("B", "champion_S1_s0", "E324s")}
mn = np.array([P["frames"][k]["sys"]["champion_S1_s0"]["man_n"] for k in range(n)], float)
valid = ~np.isnan(ct) & ~np.isnan(bt)
d = np.where(valid, ct - bt, np.nan)
pick = {}
p75 = np.nanpercentile(ct, 75)
pick["01_typical_good"] = (int(np.nanargmin(np.abs(ct - p75))), "a typical good frame: the champion's topo F1 nearest to its 75th percentile (%.3f)" % p75)
pick["02_largest_gain_over_B"] = (int(np.nanargmax(d)), "the frame where the champion gains most over the plain head")
pick["03_largest_loss_against_B"] = (int(np.nanargmin(d)), "the frame where the champion loses most against the plain head (an issue)")
c = [k for k in range(n) if mn[k] >= 4]
pick["04_maneuvers_missing"] = (min(c, key=lambda k: (ok["champion_S1_s0"][k] / mn[k], -mn[k])), "the frame with the lowest share of realised maneuvers (at least 4 GT maneuvers)")
c2 = [k for k in c if ok["B"][k] > ok["champion_S1_s0"][k]]
pick["05_junction_worse_than_B"] = (max(c2, key=lambda k: ok["B"][k] - ok["champion_S1_s0"][k]), "the frame where the champion realises fewer maneuvers than the plain head (the junction issue)")
pick["06_worst_frame"] = (int(np.nanargmin(ct)), "the frame with the lowest topo F1 of the champion")
spec, stats = [], {}
for name, (k, why) in pick.items():
    spec.append({"name": name, "k": int(k), "title": why})
    stats[name] = {"k": int(k), "champion_topo": round(float(ct[k]), 4), "B_topo": round(float(bt[k]), 4), "E324s_topo": round(float(et[k]), 4),
                   "man_B/champ/E324s of n": [int(ok["B"][k]), int(ok["champion_S1_s0"][k]), int(ok["E324s"][k]), int(mn[k])]}
tot = {s: [int(ok[s].sum()), int(mn.sum())] for s in ok}
counts = {"frames": int(valid.sum()), "champion_better_than_B_by_0.02": int(np.nansum(d > 0.02)), "champion_worse_than_B_by_0.02": int(np.nansum(d < -0.02)),
          "within_0.02": int(np.nansum(np.abs(d) <= 0.02)), "median_gain_over_B": round(float(np.nanmedian(d)), 4), "maneuvers_realised_ok_of_n": tot,
          "frames_with_fewer_maneuvers_than_B": int(np.nansum((mn >= 1) & (ok["B"] > ok["champion_S1_s0"]))), "frames_with_maneuvers": int((mn >= 1).sum())}
json.dump(spec, open(f"{H17}/spec_champion_q1117.json", "w"), indent=1)
json.dump({"picks": stats, "counts": counts}, open(f"{H17}/stats_champion_q1117.json", "w"), indent=1)
print(json.dumps(stats, indent=1)); print(json.dumps(counts, indent=1))
