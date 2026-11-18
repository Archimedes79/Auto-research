"""Loop 17: the single source of every number of the report. Reads the result pickles of E130 - E134 (and the map builder E138 when present), scores clean / failure phases with the loop-16 card and the paired scene bootstraps,
reads the route-success JSONs of E135 / E136, writes out/loop2/e17/results_master.json (the report builder reads only this file).
Usage (gcar): p17_master.py"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p17_compare as CMP  # noqa: E402

E17 = CMP.E17
PERT3 = ["dark", "blur", "cam_front3_off"]
PERT4 = ["fog", "radar_off", "extr3", "cam_front_off"]


def load_any(spec, pert):
    if spec.startswith("oof:"):
        d = pickle.load(open(f"{E17}/{spec[4:]}.pkl", "rb"))
        return dict(rows=np.asarray(d["rows"]), scene=np.asarray(d["scene"]), fail=np.asarray(d["fail"]), graphs=d["graphs"], comp=np.asarray(d["comp"]))
    return CMP.load(spec, pert)


def entries(p):
    e = {}
    if p in PERT3:
        e["ctrl_single"] = "control:single"
        e["ctrl_hold"] = "control:maxdecay_0.95"
        e["ctrl_ema"] = "control:ema_hold_mon_0.5"
        e["champ_single"] = f"builder:e130_champion/{p}_single"
        e["champ_hold"] = f"builder:e131_hold/{p}_maxdecay_0.95"
        e["champ_ema"] = f"builder:e131_hold/{p}_ema_hold_mon_0.5"
        e["norec_single"] = f"builder:e132_norec/{p}_single"
        e["norec_hold"] = f"builder:e132_norec/{p}_maxdecay_0.95"
        e["ftall_single"] = f"builder:e133a_ftall/{p}_single_ftall"
        e["ftall_hold"] = f"builder:e133a_ftall/{p}_maxdecay_0.95_ftall"
        for v, k in (("single", "ft_single"), ("maxdecay_0.95", "ft_hold"), ("ema_hold_mon_0.5", "ft_ema")):
            e[k] = f"oof:e133b_oof_{p}_{v}"
        e["gate"] = f"builder:e134_gate/{p}_gate"
        for mode in ("m0", "m2", "mnone", "mwrong"):
            for v, k in (("single", "map_single"), ("maxdecay_0.95", "map_hold")):
                if os.path.exists(f"{E17}/e138_map/{p}_{v}{'_' + mode}.pkl"):
                    e[f"{k}_{mode}"] = f"builder:e138_map/{p}_{v}_{mode}"
    else:
        e["champ_single"] = f"builder:e130b_more/{p}_single_e37"
        e["champ_hold"] = f"builder:e130b_more/{p}_maxdecay_0.95_e37"
    return e


PAIRS = [("champ_single", "ctrl_single"), ("champ_hold", "champ_single"), ("champ_hold", "ctrl_hold"), ("champ_ema", "champ_single"), ("norec_hold", "champ_hold"), ("ftall_hold", "champ_hold"), ("ftall_single", "champ_single"),
         ("ft_single", "champ_single"), ("ft_hold", "champ_hold"), ("gate", "norec_hold"), ("gate", "champ_hold"), ("gate", "ctrl_hold"), ("map_single_m0", "champ_single"), ("map_hold_m0", "champ_hold"), ("map_hold_m2", "champ_hold"), ("map_single_mwrong", "champ_single"), ("map_single_mnone", "champ_single")]


def main():
    out = dict(perts={})
    for p in PERT3 + PERT4:
        ent = {k: load_any(v, p) for k, v in entries(p).items()}
        res = dict(cards={}, boot={})
        for k, d in ent.items():
            cs = CMP.cards(d)
            res["cards"][k] = {ph: {m: float(c[m]) for m in ("S", "F1@1.0", "F1@0.5", "J", "vert100", "false_len_frac", "lanes_found", "R@1.0", "P@1.0")} for ph, c in cs.items()}
            res["cards"][k]["n"] = dict(cl=int((~d["fail"]).sum()), fail=int(d["fail"].sum()))
        for a, b in PAIRS:
            if a in ent and b in ent:
                b0, lo, hi, nf, ns = CMP.boot(ent[a], ent[b])
                res["boot"][f"{a} - {b}"] = dict(diff=float(b0), lo=float(lo), hi=float(hi), frames=nf, scenes=ns)
        out["perts"][p] = res
        print("done", p, flush=True)
    # route success (E135 / E135b / E136)
    route = {}
    for d in ("e135_route", "e136_route", "e138_route"):
        dd = f"{E17}/{d}"
        if os.path.isdir(dd):
            for f in sorted(os.listdir(dd)):
                if f.startswith("route_") and f.endswith(".json"):
                    j = json.load(open(f"{dd}/{f}"))
                    cell = {}
                    for k, v in j.items():
                        if isinstance(v, dict) and "route_success" in v:
                            cell[k] = dict(route_success=v["route_success"], n=v["n"])
                    route[f"{d}/{f[6:-5]}"] = cell
    out["route"] = route
    json.dump(out, open(f"{E17}/results_master.json", "w"), indent=1)
    print("written", f"{E17}/results_master.json")


if __name__ == "__main__":
    main()
