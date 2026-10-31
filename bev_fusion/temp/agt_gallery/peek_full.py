"""scratch: numbers for the refreshed overview text (full scan)"""
import collections, json
S = json.load(open("temp/agt_gallery/scan_15efbde2d1.json"))
bad = [r for r in S if r["ego_road"] < 0.5]
bs = collections.Counter(r["scene"] for r in bad)
loc = {r["scene"]: r["loc"] for r in S}
print("ego off map", len(bad), "of", len(S), len(bad) / len(S), "sequences", len(bs), [(s, n, loc[s]) for s, n in sorted(bs.items())])
off = collections.Counter()
for r in S:
    off[r["scene"]] += r["mov_veh"] - r["mov_veh_road"]
tot = sum(off.values()); allmov = sum(r["mov_veh"] for r in S)
print("moving off road", tot, "of", allmov, tot / allmov, "top10 share", sum(v for _, v in off.most_common(10)) / tot)
print([(s, n, loc[s]) for s, n in off.most_common(12)])
for lc in sorted(set(loc.values())):
    sub = [r for r in S if r["loc"] == lc]
    o = collections.Counter()
    for r in sub:
        o[r["scene"]] += r["mov_veh"] - r["mov_veh_road"]
    t = sum(o.values())
    print(lc, "off", t, "top3", [(s, n) for s, n in o.most_common(3)], "share top3", sum(n for _, n in o.most_common(3)) / max(1, t))
fl = collections.Counter()
for r in S:
    fl.update(r["flags"])
n_obs = sum(r["veh"] + r["ped"] + r["two"] for r in S)
print("flags", dict(fl), "obs", n_obs)
print("n_shift", sum(r["n_shift"] for r in S), "shift_max cm", max(r["shift_max"] for r in S) * 100)
print(">60", sum(r["vmax"] > 60 for r in S), ">80", sum(r["vmax"] > 80 for r in S), "mean obs/kf", n_obs / len(S), "mov/kf", allmov / len(S))
print("ign share", sum(r["n_ign"] for r in S) / (sum(r["n_ign"] for r in S) + n_obs))
T = json.load(open("temp/agt_gallery/tables.json"))
print("night rows seq", T["night"]["seq"], "kf", T["night"]["kf"], "rain seq", T["rain"]["seq"])
# night only in Singapore? by location
print(collections.Counter(r["loc"] for r in S if r["night"]))
print(collections.Counter(r["loc"] for r in S if r["rain"]))
# ped_ok for night
