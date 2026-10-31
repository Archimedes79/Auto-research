"""scratch: map holes and a few more numbers from scan_<sig>.json"""
import collections
import json
import sys

S = json.load(open(f"temp/agt_gallery/scan_{sys.argv[1]}.json"))
n = len(S)
bad = [r for r in S if r["ego_road"] < 0.5]
print("keyframes", n, "ego off the map road:", len(bad), f"({len(bad) / n:.2%})", "sequences:", len({r['scene'] for r in bad}))
c = collections.Counter(r["scene"] for r in bad)
print("worst sequences:", c.most_common(8))
by_loc = collections.Counter(r["loc"] for r in bad)
print("by location:", dict(by_loc), {l: sum(1 for r in S if r["loc"] == l) for l in by_loc})
empty = [r for r in S if r["drivable"] < 0.02]
print("drivable layer (almost) empty:", len(empty), "keyframes in", len({r['scene'] for r in empty}), "sequences")
whole = [sc for sc in c if c[sc] >= 0.8 * sum(1 for r in S if r["scene"] == sc)]
print("sequences with the ego off the map in >= 80% of their sampled keyframes:", whole)
print("moving vehicles total", sum(r["mov_veh"] for r in S), "off the drivable area", sum(r["mov_veh"] - r["mov_veh_road"] for r in S))
off = collections.Counter()
for r in S:
    off[r["scene"]] += r["mov_veh"] - r["mov_veh_road"]
print("off-road moving vehicles concentrate in:", off.most_common(8), "share of the top 10 sequences:",
      sum(v for _, v in off.most_common(10)) / max(1, sum(off.values())))
print("n_shift", sum(r["n_shift"] for r in S), "max shift (m)", max(r["shift_max"] for r in S))
fl = collections.Counter()
for r in S:
    fl.update(r["flags"])
print("flags", dict(fl))
print("keyframes with speed > 60 km/h:", sum(r["vmax"] > 60 for r in S), "> 80:", sum(r["vmax"] > 80 for r in S))
print("per keyframe: observed boxes mean", sum(r["veh"] + r["ped"] + r["two"] for r in S) / n, "moving vehicles mean", sum(r["mov_veh"] for r in S) / n,
      "ignored mean", sum(r["n_ign"] for r in S) / n)
