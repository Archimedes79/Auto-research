"""scratch (gallery of the GT quality overview): aggregate scan_<sig>.json and pick the examples.

  analyse.py SIGNATURE  -> prints the tables, writes temp/agt_gallery/tables.json and temp/agt_gallery/picks.json
"""
import collections
import json
import random
import sys

import numpy as np

sig = sys.argv[1]
S = json.load(open(f"temp/agt_gallery/scan_{sig}.json"))
rng = random.Random(1)
print("keyframes", len(S), "sequences", len({r["scene"] for r in S}))


def agg(rows):
    """quality numbers of a group of keyframes (sums over the keyframes, ratios of sums)."""
    a = lambda k: sum(r[k] for r in rows)                                                       # noqa: E731
    fl = collections.Counter()
    for r in rows:
        fl.update(r["flags"])
    n_obs = a("veh") + a("ped") + a("two")
    return dict(kf=len(rows), seq=len({r["scene"] for r in rows}), obs=n_obs, veh=a("veh"), ped=a("ped"), two=a("two"),
                ign_share=a("n_ign") / max(1, a("n_ign") + n_obs),
                mov_veh=a("mov_veh"), mov_veh_road=a("mov_veh_road") / max(1, a("mov_veh")),
                ped_ok=a("ped_ok") / max(1, a("ped")), off_plane=a("off_plane") / max(1, a("in40")),
                vis4_occ=a("vis4_occ") / max(1, a("vis4")), ego_road=float(np.mean([r["ego_road"] for r in rows])),
                ego_bad=float(np.mean([r["ego_road"] < 0.5 for r in rows])), camera=float(np.mean([r["camera"] for r in rows])),
                cl_road=a("centreline_road") / max(1, a("centreline")),
                unknown=fl["short_track"] / max(1, n_obs), override=fl["radar_override"] / max(1, n_obs), conflict=fl["radar_conflict"] / max(1, n_obs),
                vmax=max(r["vmax"] for r in rows))


groups = {"all": S}
for loc in sorted({r["loc"] for r in S}):
    groups[loc] = [r for r in S if r["loc"] == loc]
groups["night"] = [r for r in S if r["night"]]
groups["day"] = [r for r in S if not r["night"]]
groups["rain"] = [r for r in S if r["rain"]]
print({k: len(v) for k, v in groups.items()})
tables = {k: agg(v) for k, v in groups.items() if v}
json.dump(tables, open("temp/agt_gallery/tables.json", "w"), indent=1)
hdr = ["kf", "seq", "obs", "veh", "ped", "two", "ign_share", "mov_veh", "mov_veh_road", "ped_ok", "off_plane", "vis4_occ", "ego_bad", "cl_road",
       "unknown", "override", "conflict", "vmax"]
print(f"{'group':26s}" + "".join(f"{h:>13s}" for h in hdr))
for k, t in tables.items():
    print(f"{k:26s}" + "".join(f"{t[h]:13.3f}" if isinstance(t[h], float) else f"{t[h]:13d}" for h in hdr))

picks, used = [], set()


def pick(rows, key, n, title, note, reverse=True, distinct_scene=True, min_gap=6):
    got = []
    for r in sorted(rows, key=key, reverse=reverse):
        if r["scene"] in used:
            continue
        if len(got) >= n:
            break
        used.add(r["scene"])
        got.append(r)
        picks.append(dict(scene=r["scene"], k=r["k"], tok=r["tok"], title=title, note=note, loc=r["loc"], desc=r["desc"]))
    return got


# 1 typical: a seeded random draw (no cherry picking) - one keyframe of 9 different random sequences
seqs = sorted({r["scene"] for r in S})
for sc in rng.sample(seqs, 9):
    rows = [r for r in S if r["scene"] == sc]
    r = rng.choice(rows)
    used.add(sc)
    picks.append(dict(scene=r["scene"], k=r["k"], tok=r["tok"], title="typical (random draw)", note="seeded random sequence and keyframe, not selected", loc=r["loc"], desc=r["desc"]))
busy = lambda r: r["veh"] + r["ped"] + r["two"]                                                  # noqa: E731
for loc in sorted({r["loc"] for r in S}):
    pick([r for r in S if r["loc"] == loc], lambda r: (min(busy(r), 40), r["mov_veh"]), 1, f"location {loc}", "busy keyframe of this map", True)
pick([r for r in S if r["night"]], lambda r: busy(r), 2, "night", "darkest conditions for the camera; GT does not use the image")
pick([r for r in S if r["rain"] and not r["night"]], lambda r: busy(r), 2, "rain", "wet road, spray, wipers")
pick([r for r in S if r["ped"] >= 5], lambda r: (r["ped"], r["ped_ok"]), 2, "pedestrians", "crowds at crossings / walkways")
pick(S, lambda r: r["vmax"], 2, "fast traffic", "highest speeds of the dataset")
pick([r for r in S if "park" in r["desc"].lower()], lambda r: r["veh"], 1, "parking", "many stationary boxes (one fixed world pose each)")
pick([r for r in S if r["two"] >= 2], lambda r: r["two"], 1, "two-wheelers", "bicycles / motorcycles")
pick([r for r in S if "construction" in r["desc"].lower()], lambda r: busy(r), 1, "construction", "unusual layout")
# 2 failure and limit cases
pick(S, lambda r: r["flags"].get("radar_conflict", 0), 2, "limit: radar conflict", "radar and track velocity disagree; track kept (flag !)")
pick(S, lambda r: r["flags"].get("radar_override", 0), 2, "limit: radar override", "the radar fixed the track speed (flag o)")
pick(S, lambda r: r["flags"].get("short_track", 0), 1, "limit: unknown velocity", "short track: velocity unknown, weight 0, box not moved to camera time")
pick(S, lambda r: r["off_plane"] / max(1, r["in40"]) if r["in40"] > 8 else 0, 2, "limit: off the ground plane", "hills / overpasses: box bottoms far from the ego ground plane")
pick(S, lambda r: r["vis4_occ"], 2, "limit: occlusion mask", "annotators rate boxes 80-100 % visible but the 2D mask hides them")
pick(S, lambda r: r["mov_veh"] - r["mov_veh_road"], 2, "limit: moving car off the drivable area", "map / annotation mismatch")
pick(S, lambda r: r["n_ign"], 1, "limit: many ignored boxes", "boxes without any lidar / radar point (yellow)")
pick(S, lambda r: -r["ego_road"], 1, "limit: ego not on the drivable area", "map / localisation error under the ego car")
thumbs = []
pool = [r for r in S]
for r in rng.sample(pool, 96):
    thumbs.append(dict(scene=r["scene"], k=r["k"], tok=r["tok"]))
json.dump(dict(examples=picks, thumbs=thumbs), open("temp/agt_gallery/picks.json", "w"), indent=1)
print(len(picks), "examples,", len(thumbs), "thumbnails")
for p in picks:
    print(f"{p['title'][:42]:42s} {p['scene']} k{p['k']:2d} {p['loc'][:22]:22s} {p['desc'][:70]}")
