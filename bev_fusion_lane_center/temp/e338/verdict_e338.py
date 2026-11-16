"""E338 verdict by hand (the rule of the card, written before the numbers): D = sc - s per seed, KEEP / TIE / NEGATIVE INTERACTION, the additive comparison, the controls (curves, base blocks, the reproduction on B's dump).
usage: wsl bash tools/env/_py14.sh ../temp/e338/verdict_e338.py"""
import json
import os
import re

ROOT = os.environ["BFLC_ROOT"]
R = ROOT + "/runs/"
K = ("topo_F1", "dup_frac", "F1@1.0", "lanes_found", "junction_F1", "vert100_dp")
ADD = {0: dict(topo_F1=0.7046, dup_frac=0.1691, **{"F1@1.0": 0.8742}, junction_F1=0.3414, vert100_dp=18.85), 1: dict(topo_F1=0.7030, dup_frac=0.1588, **{"F1@1.0": 0.8792}, junction_F1=0.3332, vert100_dp=19.34)}
WIN = dict(topo_F1=0.003, dup_frac=0.003, **{"F1@1.0": 0.003}, junction_F1=0.01, vert100_dp=0.3)


def card(s, name, run=None):
    return json.load(open(f"{R}{run or f'e324_s_s{s}'}/{name}"))


def interval(path):
    t = open(path).read()
    m = re.search(r"topo_F1\s+A - B: point ([+-][0-9.]+), bootstrap mean [+-][0-9.]+, 95 % \[([+-][0-9.]+), ([+-][0-9.]+)\]", t)
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


tokens = []
for s in (0, 1):
    cs, csc = card(s, "ends_card_snap_thr0.1.json"), card(s, "ends_card_e338_snap_thr0.1.json")
    s_, sc = cs["numbers"]["head"], csc["numbers"]["head"]
    c2card = card(s, "ends_card_c2_snap_thr0.1.json", f"e312_p_s{s}")
    ctl = card(s, "ends_card_e338b_snap_thr0.1.json")
    print(f"== seed {s}")
    ok_ctl = all(abs(ctl["numbers"]["head"][k] - s_[k]) < 1e-6 for k in s_ if k in ctl["numbers"]["head"]) and ctl["numbers"]["base"] == cs["numbers"]["base"]
    ok_curves = csc["curves"] == cs["curves"]
    ok_base = csc["numbers"]["base"] == c2card["numbers"]["base"]
    print(f"  controls: (a) e338b reproduces s: {ok_ctl}; (b) curves sc {csc['curves']} = s {cs['curves']}: {ok_curves}; (c) base block of sc = base block of the c2 card: {ok_base}")
    d = {k: sc[k] - s_[k] for k in K}
    for k in K:
        if k not in ADD[s]:
            print(f"  {k:12s} sc {sc[k]:8.4f}  s {s_[k]:8.4f}  D {d[k]:+.4f}  | (no pre-registered number)")
            continue
        a = ADD[s][k]
        print(f"  {k:12s} sc {sc[k]:8.4f}  s {s_[k]:8.4f}  D {d[k]:+.4f}  | additive {a:8.4f}  sc - additive {sc[k] - a:+.4f}  (window +-{WIN[k]})  {'HIT' if abs(sc[k] - a) <= WIN[k] else 'miss'}")
    pt, lo, hi = interval(f"{ROOT}/temp/e338/boot_sc_vs_s_s{s}.log")
    guards = dict(F1=-d["F1@1.0"] <= 0.01, dup=d["dup_frac"] <= 0.01, vert=d["vert100_dp"] <= 1.0, junc=-d["junction_F1"] <= 0.03)
    print(f"  bootstrap D topo F1 {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]; guards {guards}")
    if not (ok_ctl and ok_curves and ok_base):
        tokens.append("INVALID")
    elif d["topo_F1"] < -0.003:
        tokens.append("NEGATIVE")
    elif d["topo_F1"] >= 0.005 and lo > 0 and all(guards.values()):
        tokens.append("KEEP")
    else:
        tokens.append("TIE")
print("per seed:", tokens)
print("TOKEN:", "INVALID (plumbing control failed)" if "INVALID" in tokens else "NEGATIVE INTERACTION" if "NEGATIVE" in tokens else "KEEP as a stack candidate" if tokens == ["KEEP", "KEEP"] else "TIE, kept (separate candidates)")
