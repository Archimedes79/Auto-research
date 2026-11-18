"""E354: the verdict table by the written rule of the card (section 0), for both seeds. Reads the stored cards only; prints the numbers and each clause."""
import json, os

M = ("topo_F1", "topo_P", "junction_F1", "lanes_found", "dup_frac", "F1@1.0", "vert100_dp")


def card(run, snap):
    p = f"runs/{run}/ends_card_{'snap_' if snap else ''}thr0.1.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))["numbers"]["head"]


def flip(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return d


out = {}
for S in (0, 1):
    runs = dict(c=f"e341_p_s{S}", z=f"e354_z_s{S}", r=f"e354_r_s{S}", off=f"e354_r_s{S}__off", shuf=f"e354_r_s{S}__shuffle", trunc=f"e354_r_s{S}__trunc")
    for ro, snap in (("unsnapped", False), ("snapped", True)):
        cs = {k: card(v, snap) for k, v in runs.items()}
        out[(S, ro)] = cs
        print(f"== seed {S}, {ro}, VAL p >= 0.1")
        for k, c in cs.items():
            print(f"  {k:6s}", "missing" if c is None else " ".join(f"{m}={c[m]:.4f}" for m in M))
        if cs["r"] and cs["z"]:
            dz = {m: cs["r"][m] - cs["z"][m] for m in M}
            print("  r - z  ", " ".join(f"{m}={dz[m]:+.4f}" for m in M))
        if cs["r"] and cs["c"]:
            dc = {m: cs["r"][m] - cs["c"][m] for m in M}
            print("  r - E341", " ".join(f"{m}={dc[m]:+.4f}" for m in M))
    fl = {"r vs z": flip(f"runs/e354_r_s{S}/flip_e354_z_s{S}.json"), "shuffle vs r": flip(f"runs/e354_r_s{S}__shuffle/flip_e354_r_s{S}.json"), "off vs r": flip(f"runs/e354_r_s{S}__off/flip_e354_r_s{S}.json")}
    print(f"== seed {S}, pair flip rate (true pairs delivered in exactly one of two consecutive keyframes / in either)")
    for k, d in fl.items():
        if d is None:
            print("  ", k, "missing"); continue
        print(f"   {k:13s} control {d['control']['flip_rate']:.4f} ({d['control']['true_pairs_t0_delivered']} delivered), run {d['run']['flip_rate']:.4f} ({d['run']['true_pairs_t0_delivered']}), run matched {d['run_matched']['flip_rate']:.4f}; "
              f"matched difference {d['matched_delivery']['difference']*100:+.2f} points [{d['matched_delivery']['lo95']*100:+.2f}, {d['matched_delivery']['hi95']*100:+.2f}]; equal threshold {d['equal_threshold']['difference']*100:+.2f} [{d['equal_threshold']['lo95']*100:+.2f}, {d['equal_threshold']['hi95']*100:+.2f}]")


def ok(cs, key):
    return cs["r"] is not None and cs["z"] is not None and cs["c"] is not None


print("\n== the clauses of the card (r against z; E341 for the reference clause)")
res = {}
for ro in ("unsnapped", "snapped"):
    for S in (0, 1):
        cs = out[(S, ro)]
        if not ok(cs, ro):
            print(ro, S, "incomplete"); continue
        r, z, c = cs["r"], cs["z"], cs["c"]
        res[(S, ro)] = dict(gain=r["topo_F1"] - z["topo_F1"], vs_c=r["topo_F1"] - c["topo_F1"],
                            guards=dict(F1_1=r["F1@1.0"] - z["F1@1.0"] >= -0.01, dup=r["dup_frac"] - z["dup_frac"] <= 0.01, vert=r["vert100_dp"] - z["vert100_dp"] <= 1.0, junc=r["junction_F1"] - z["junction_F1"] >= -0.03))
for k, v in sorted(res.items()):
    print(k, "gain over z %+.4f" % v["gain"], "| vs E341 %+.4f" % v["vs_c"], "| guards", v["guards"])
if len(res) == 4:
    un = [res[(S, "unsnapped")] for S in (0, 1)]
    sn = [res[(S, "snapped")] for S in (0, 1)]
    keep = all(u["gain"] >= 0.01 for u in un) and all(s["gain"] >= -0.005 for s in sn) and all(all(v["guards"].values()) for v in res.values()) and all(v["vs_c"] >= -0.01 for v in res.values())
    tie = (not keep) and all(v["gain"] >= -0.01 for v in res.values()) and all(v["vs_c"] >= -0.01 for v in res.values()) and all(all(v["guards"].values()) for v in res.values())
    cut = all(u["gain"] < -0.01 for u in un) or all(any(not g for g in v["guards"].values()) for v in res.values())
    print("KEEP" if keep else "", "TIE" if tie else "", "CUT" if cut else "")
    if not (keep or tie or cut):
        print("INCONCLUSIVE")
