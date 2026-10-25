"""Loop 17: the second results file (results_extra.json) for the report revision after the reviews: the matched clean reference (E208), the unseen failure types (E209), the map-only baseline (E210), the consumer under failure (E212),
and scene-bootstrap intervals of route success differences (the route head's seed spread is 0.05 - 0.09, so route differences need an interval).
Usage (gcar): p17_extra.py"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p17_compare as CMP  # noqa: E402
import p17_master as PM  # noqa: E402

E17 = CMP.E17
HOME = os.path.expanduser("~")
PERT3 = ["dark", "blur", "cam_front3_off"]


def ex(path):
    return os.path.exists(f"{E17}/{path}.pkl")


def cards_of(spec, p):
    d = PM.load_any(spec, p)
    cs = CMP.cards(d)
    return d, {ph: {m: float(c[m]) for m in ("S", "F1@1.0", "F1@0.5", "J", "vert100", "false_len_frac", "lanes_found", "P@1.0", "R@1.0")} for ph, c in cs.items()}


def route_boot(tag_a, tag_b, series, draws=2000, seed=0):
    """scene bootstrap of the route-success difference a - b on the failure-half frames (mean of the two route-head seeds, map error 0)"""
    ja = json.load(open(f"{E17}/{tag_a}.json"))
    jb = json.load(open(f"{E17}/{tag_b}.json"))
    meta = np.load(f"{HOME}/exp_gcar_b123/p17/{series}.meta.npz")
    ph, sc = meta["phase"], meta["scene"]

    def vec(j):
        xs = [np.array([np.nan if v is None else v for v in c["per_frame"]], float) for k, c in j.items() if isinstance(c, dict) and c.get("per_frame") and k.startswith("sigma0_") and k.endswith("_fail")]
        return np.nanmean(np.stack(xs), 0)

    a, b = vec(ja), vec(jb)
    ok = (ph == 1) & np.isfinite(a) & np.isfinite(b)
    a, b, s = a[ok], b[ok], sc[ok]
    u = np.unique(s)
    rng = np.random.RandomState(seed)
    d0 = float(a.mean() - b.mean())
    ds = []
    for _ in range(draws):
        pick = rng.choice(u, len(u))
        idx = np.concatenate([np.where(s == x)[0] for x in pick])
        ds.append(a[idx].mean() - b[idx].mean())
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(diff=d0, lo=float(lo), hi=float(hi), n=int(ok.sum()), scenes=int(len(u)), mean_a=float(a.mean()), mean_b=float(b.mean()))


def main():
    out = dict(cleanref={}, unseen={}, route_boot={}, maponly=None, consumer={})
    # E208 matched clean reference: the failure-half frames of the series without any perturbation
    for p in PERT3:
        r = {}
        for key, spec in (("champ_single", f"builder:e141_clean/{p}_single_cleanref"), ("champ_hold", f"builder:e141_clean/{p}_maxdecay_0.95_cleanref"), ("gate", f"builder:e141_clean/{p}_gate_cleanref"),
                          ("B_single", f"builder:e141_cleanB/{p}_single_cleanref_m0"), ("B_hold", f"builder:e141_cleanB/{p}_maxdecay_0.95_cleanref_m0"), ("gateB", f"builder:e141_cleanB/{p}_gateB_cleanref_m0"),
                          ("C_single", f"builder:e141_cleanB/{p}_single_cleanref_C"), ("C_hold", f"builder:e141_cleanB/{p}_maxdecay_0.95_cleanref_C")):
            path = spec.split(":", 1)[1]
            if ex(path):
                _, c = cards_of(spec, p)
                r[key] = c["fail"]
        out["cleanref"][p] = r
    # E209 unseen failure types: builder trained WITHOUT night and blur rows, evaluated on night and blur
    for p in ("dark", "blur"):
        r = dict(cards={}, boot={})
        ent = {}
        for key, spec in (("B2_single", f"builder:e142/{p}_single_B2m0"), ("B2_hold", f"builder:e142/{p}_maxdecay_0.95_B2m0"), ("B2none_single", f"builder:e142/{p}_single_B2none"), ("B2none_hold", f"builder:e142/{p}_maxdecay_0.95_B2none"),
                          ("C2_single", f"builder:e142/{p}_single_C2none"), ("C2_hold", f"builder:e142/{p}_maxdecay_0.95_C2none"), ("champ_single", f"builder:e132_norec/{p}_single"), ("champ_hold", f"builder:e132_norec/{p}_maxdecay_0.95"),
                          ("B_single", f"builder:e139_b/{p}_single_m0"), ("B_hold", f"builder:e139_b/{p}_maxdecay_0.95_m0"), ("C_single", f"builder:e139_c/{p}_single_mnone"), ("C_hold", f"builder:e139_c/{p}_maxdecay_0.95_mnone")):
            path = spec.split(":", 1)[1]
            if ex(path):
                d, c = cards_of(spec, p)
                ent[key] = d
                r["cards"][key] = c["fail"]
        for a, b in (("B2_single", "champ_single"), ("B2_hold", "champ_hold"), ("C2_single", "champ_single"), ("C2_hold", "champ_hold"), ("B2_single", "C2_single"), ("B2_hold", "C2_hold"), ("B2none_hold", "C2_hold"), ("B2_hold", "B2none_hold"), ("B2_single", "B_single"), ("B2_hold", "B_hold")):
            if a in ent and b in ent:
                b0, lo, hi, nf, ns = CMP.boot(ent[a], ent[b])
                r["boot"][f"{a} - {b}"] = dict(diff=float(b0), lo=float(lo), hi=float(hi), frames=nf, scenes=ns)
        out["unseen"][p] = r
    # E210 map-only baseline, E212 consumer
    if os.path.exists(f"{E17}/e143_maponly.json"):
        out["maponly"] = json.load(open(f"{E17}/e143_maponly.json"))
    for f in ("e144_champ", "e144_map"):
        if os.path.exists(f"{E17}/{f}/consumer.json"):
            out["consumer"][f] = json.load(open(f"{E17}/{f}/consumer.json"))
    # route-success intervals (mean of two seeds, map error 0)
    pairs = []
    for p in PERT3:
        pairs += [(f"e135_route/{p}_single", f"e135_route/{p}_maxdecay_0.95", f"{p}_single", f"{p}: hold - single frame (champion; hold = recursion off)"),
                  (f"e135_route/{p}_gate", f"e135_route/{p}_maxdecay_0.95", f"{p}_gate", f"{p}: gate - hold"),
                  (f"e139_route/{p}_maxdecay_0.95_B", f"e135_route/{p}_maxdecay_0.95", f"{p}_maxdecay_0.95", f"{p}: B + hold - champion + hold"),
                  (f"e139_route/{p}_maxdecay_0.95_B", f"e139_route/{p}_maxdecay_0.95_C", f"{p}_maxdecay_0.95", f"{p}: B + hold - C + hold"),
                  (f"e139_route/{p}_single_B", f"e139_route/{p}_single_C", f"{p}_single", f"{p}: B single - C single"),
                  (f"e139_route/{p}_maxdecay_0.95_C", f"e135_route/{p}_maxdecay_0.95", f"{p}_maxdecay_0.95", f"{p}: C + hold - champion + hold"),
                  (f"e140_route/{p}_gateB_B", f"e139_route/{p}_maxdecay_0.95_B", f"{p}_gateB", f"{p}: B + gate - B + hold"),
                  (f"e140_route/{p}_gateB_B", f"e135_route/{p}_gate", f"{p}_gateB", f"{p}: B + gate - champion + gate"),
                  (f"e139_route/{p}_maxdecay_0.95_B", f"e139_route/{p}_single_B", f"{p}_maxdecay_0.95", f"{p}: B hold - B single")]
    for ta, tb, ser, lab in pairs:
        if os.path.exists(f"{E17}/{ta}_B.json" if False else f"{E17}/{ta}.json") and os.path.exists(f"{E17}/{tb}.json"):
            try:
                out["route_boot"][lab] = route_boot(ta, tb, ser)
            except Exception as e:  # missing per_frame in an old file
                out["route_boot"][lab] = dict(error=str(e))
    json.dump(out, open(f"{E17}/results_extra.json", "w"), indent=1)
    print("written results_extra.json", list(out["route_boot"].keys())[:3])


if __name__ == "__main__":
    main()
