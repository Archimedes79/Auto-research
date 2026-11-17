"""Report analysis (CPU, from stored per-frame counts; lane lead's order of 2026-11-15): (a) the OWN-LANE quality (the tier metrics of E276: tier 1 = the ego's own lane, tier 2 = the lanes next to it that the ego could use, tier 3 = the rest; t1_R_1 = recall of the VISIBLE
tier-1 GT points within 1.0 m of a delivered point, z1_P_1 = precision of the delivered points whose nearest GT point (within 2.5 m) is tier 1, t12_F1_1 = F1 of tiers 1 + 2) and (b) numbers per situation (far range, junction frames, crowded junctions, night / rain scenes by the nuScenes scene description) of B (E273 + close-node
step, the plain graph), E312 snapped, E324 s snapped and PRIMARY, both seeds, VAL 873 frames, paired scene bootstrap.
usage: wsl bash tools/env/_py14.sh own_lane.py   -> temp/report/own_lane.json, temp/report/own_lane_table.md"""
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.getcwd()
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import qgraph_eval as QE  # noqa: E402

OUT = os.path.join(ROOT, "temp", "report")
SYS = (("B", "e312_p_s", "ends_card_snap_thr0.1", "base"), ("E312 snapped", "e312_p_s", "ends_card_snap_thr0.1", "head"), ("E324 s snapped", "e324_s_s", "ends_card_snap_thr0.1", "head"),
       ("E324 s PRIMARY", "e324_s_s", "ends_card_snap_nogeom_attach_thr0.1", "head"))
NB = 1000


def load(prefix, card, which, seed):
    c = pickle.load(open(os.path.join(ROOT, "runs", f"{prefix}{seed}", card + ".counts.pkl"), "rb"))
    return c["scenes"], c[which]


def tier_arrays(frames):
    rec = np.array([f["tier"]["rec"] if f.get("tier") is not None else np.zeros((3, 4)) for f in frames], float)
    prec = np.array([f["tier"]["prec"] if f.get("tier") is not None else np.zeros((4, 2)) for f in frames], float)
    has = np.array([f.get("tier") is not None for f in frames])
    return rec, prec, has


def t_numbers(rec, prec):
    n = QE.tier_numbers(dict(rec=rec.sum(0), prec=prec.sum(0)))
    r1, p1 = n["t1_R_1"], n["z1_P_1"]
    n["t1_F1_1"] = 2 * r1 * p1 / (r1 + p1) if r1 + p1 > 0 else 0.0
    return {k: n[k] for k in ("t1_R_1", "z1_P_1", "t1_F1_1", "t2_R_1", "z2_P_1", "t12_R_1", "t12_P_1", "t12_F1_1", "t3_R_1", "t1_len")}


def scene_names():
    splits = json.load(open(os.path.join(ROOT, "data", "splits.json")))
    rows = splits["val"]
    b123 = os.path.expanduser("~/exp_gcar_b123")
    tokens = json.load(open(os.path.join(b123, "feats", "val_tokens.json")))
    nu = os.path.expanduser("~/data/nuscenes_full/v1.0-trainval")
    sample = {s["token"]: s["scene_token"] for s in json.load(open(os.path.join(nu, "sample.json")))}
    scene = {s["token"]: s for s in json.load(open(os.path.join(nu, "scene.json")))}
    out = []
    for r in rows:
        t = tokens[r]
        t = t if isinstance(t, str) else t[0] if isinstance(t, (list, tuple)) else t.get("token", t)
        sc = scene[sample[t]]
        out.append((sc["name"], sc["description"]))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    names = scene_names()
    desc = [d for _, d in names]
    night = np.array(["night" in d.lower() for d in desc])
    rain = np.array([("rain" in d.lower()) or ("wet" in d.lower()) for d in desc])
    res = dict(note="VAL 873 frames; tier = l4 relevance tier of the GT lane points (E276): 1 own lane, 2 next lanes the ego could use, 3 the rest; numbers pooled over the frames of a seed; boot = paired scene bootstrap (22 scenes, 1000 resamples) 95 % interval",
               scenes={n: d for n, d in sorted(set(names))}, night_frames=int(night.sum()), rain_frames=int(rain.sum()), systems={})
    for seed in (0, 1):
        for label, prefix, card, which in SYS:
            scenes, frames = load(prefix, card, which, seed)
            sc = np.asarray(scenes)
            assert len(frames) == 873 and len(names) == 873
            rec, prec, has = tier_arrays(frames)
            rng = np.random.RandomState(seed)
            uniq = sorted(set(sc.tolist()))
            where = {s: np.where(sc == s)[0] for s in uniq}
            picks = [np.concatenate([where[uniq[i]] for i in rng.randint(0, len(uniq), len(uniq))]) for _ in range(NB)]
            point = t_numbers(rec, prec)
            boots = {k: [] for k in point}
            for pk in picks:
                n = t_numbers(rec[pk], prec[pk])
                for k in point:
                    boots[k].append(n[k])
            junc = np.array([f["lane"]["junc"][1] for f in frames], float)                       # GT junction count of the frame
            jf = junc > 0
            crowd = junc >= np.percentile(junc[jf], 75) if jf.any() else jf
            groups = {"all": np.ones(873, bool), "junction frames (GT junction nodes > 0)": jf, f"crowded junction frames (>= {np.percentile(junc[jf], 75):.0f} GT junction nodes)": crowd, "no junction": ~jf,
                      "night scenes": night, "day scenes": ~night, "rain / wet scenes": rain, "dry scenes": ~rain}
            sit = {}
            for g, m in groups.items():
                if m.sum() == 0:
                    continue
                num = QE.graph_numbers([frames[i] for i in np.where(m)[0]])
                tn = t_numbers(rec[m], prec[m])
                sit[g] = dict(frames=int(m.sum()), **{k: float(num[k]) for k in ("topo_F1", "lanes_found", "F1@1.0", "R@1_far", "junction_F1", "dup_frac") if k in num}, t1_R_1=tn["t1_R_1"], z1_P_1=tn["z1_P_1"], t1_F1_1=tn["t1_F1_1"])
            res["systems"][f"{label} s{seed}"] = dict(tier=point, tier_boot={k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}, situations=sit,
                                                      tier_boot_samples={k: [float(x) for x in v] for k, v in boots.items() if k in ("t1_R_1", "z1_P_1", "t1_F1_1", "t12_F1_1")})
    # paired differences against B per seed, same resamples (the same seed gives the same picks per label)
    res["paired_vs_B"] = {}
    for seed in (0, 1):
        b = res["systems"][f"B s{seed}"]
        for label, *_ in SYS[1:]:
            s = res["systems"][f"{label} s{seed}"]
            d = {}
            for k in ("t1_R_1", "z1_P_1", "t1_F1_1", "t12_F1_1"):
                diff = np.asarray(s["tier_boot_samples"][k]) - np.asarray(b["tier_boot_samples"][k])
                d[k] = dict(point=s["tier"][k] - b["tier"][k], lo=float(np.percentile(diff, 2.5)), hi=float(np.percentile(diff, 97.5)))
            res["paired_vs_B"][f"{label} s{seed}"] = d
    for v in res["systems"].values():
        v.pop("tier_boot_samples", None)
    json.dump(res, open(os.path.join(OUT, "own_lane.json"), "w"), indent=1)
    # the table
    L = ["# Own-lane quality and situations (VAL 873 frames, from stored per-frame counts)", "", "| system | seed | t1 recall @1 m (own lane, visible) | t1 precision | t1 F1 | tiers 1+2 F1 | far recall R@1 (> 35 m) |", "|---|---|---|---|---|---|---|"]
    for seed in (0, 1):
        for label, *_ in SYS:
            s = res["systems"][f"{label} s{seed}"]
            t = s["tier"]
            L.append(f"| {label} | {seed} | {t['t1_R_1']:.4f} [{s['tier_boot']['t1_R_1'][0]:.4f}, {s['tier_boot']['t1_R_1'][1]:.4f}] | {t['z1_P_1']:.4f} | {t['t1_F1_1']:.4f} | {t['t12_F1_1']:.4f} | {s['situations']['all'].get('R@1_far', float('nan')):.4f} |")
    L += ["", "paired differences against B (E273 + close-node step), tier-1 F1 / own-lane recall, 95 % scene interval:", ""]
    for k, d in res["paired_vs_B"].items():
        L.append(f"- {k}: t1 F1 {d['t1_F1_1']['point']:+.4f} [{d['t1_F1_1']['lo']:+.4f}, {d['t1_F1_1']['hi']:+.4f}], t1 recall {d['t1_R_1']['point']:+.4f} [{d['t1_R_1']['lo']:+.4f}, {d['t1_R_1']['hi']:+.4f}], t12 F1 {d['t12_F1_1']['point']:+.4f} [{d['t12_F1_1']['lo']:+.4f}, {d['t12_F1_1']['hi']:+.4f}]")
    L += ["", "per situation (seed 0 | seed 1): topo F1 / lanes found / far recall / junction F1 / own-lane (t1) F1", ""]
    for g in res["systems"]["B s0"]["situations"]:
        L.append(f"## {g} ({res['systems']['B s0']['situations'][g]['frames']} frames)")
        for label, *_ in SYS:
            a, b = res["systems"][f"{label} s0"]["situations"].get(g), res["systems"][f"{label} s1"]["situations"].get(g)
            if a and b:
                f = lambda s: f"{s['topo_F1']:.3f} / {s['lanes_found']:.3f} / {s.get('R@1_far', float('nan')):.3f} / {s.get('junction_F1', float('nan')):.3f} / {s['t1_F1_1']:.3f}"  # noqa: E731
                L.append(f"- {label}: {f(a)} | {f(b)}")
    open(os.path.join(OUT, "own_lane_table.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:26]))


if __name__ == "__main__":
    main()
