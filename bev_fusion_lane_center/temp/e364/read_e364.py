"""E364: the reading rules of the card on the stored cards, the bootstraps and the mask result (by hand: D87 replay).  usage: wsl bash tools/env/_py14.sh read_e364.py"""
import json
import os
import pickle
import re

ROOT = os.environ["BFLC_ROOT"]
K = ("topo_F1", "dup_frac", "F1@1.0", "junction_F1", "lanes_found", "vert100_dp")
FIELDS = ("numbers", "curves", "wired_both_ends", "long_links_dropped", "ends_within_1m_of_a_delivered_piece_end")
STORED_DROPPED = {0: 9195, 1: 8180}


def card(s, tag=None):
    name = "snap_nogeom_attach" if tag is None else f"{tag}_snap_nogeom_attach"
    return json.load(open(f"{ROOT}/runs/e324_s_s{s}/ends_card_{name}_thr0.1.json"))


def counts(s, tag=None):
    name = "snap_nogeom_attach" if tag is None else f"{tag}_snap_nogeom_attach"
    return pickle.load(open(f"{ROOT}/runs/e324_s_s{s}/ends_card_{name}_thr0.1.counts.pkl", "rb"))


def interval(path):
    t = open(path).read()
    m = re.search(r"topo_F1\s+A - B: point ([+-][0-9.]+), bootstrap mean [+-][0-9.]+, 95 % \[([+-][0-9.]+), ([+-][0-9.]+)\]", t)
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


fx_all = True
for s in (0, 1):
    p = f"{ROOT}/runs/e324_s_s{s}/ends_card_e364f_snap_nogeom_attach_thr0.1.json"
    if not os.path.exists(p):
        fx_all = False
        print(f"FIXTURE seed {s}: missing")
        continue
    f, c = card(s, "e364f"), card(s)
    ok = all(f[k] == c[k] for k in FIELDS) and repr(counts(s, "e364f")) == repr(counts(s)) and c["long_links_dropped"] == STORED_DROPPED[s]
    fx_all = fx_all and ok
    print(f"FIXTURE seed {s}: {'REPRODUCES the stored PRIMARY card (numbers, curves, wired, long links dropped, within-1m shares, counts file)' if ok else 'DIFFERS'}")
reads = []
for s in (0, 1):
    c, n = card(s), card(s, "e364n")
    hc, hn = c["numbers"]["head"], n["numbers"]["head"]
    d = {k: hn[k] - hc[k] for k in K}
    iv = interval(f"{ROOT}/temp/e364/boot_n_vs_c_s{s}.log")
    print(f"seed {s}: long links dropped c {c['long_links_dropped']} -> n {n['long_links_dropped']} (stop check: n must be 0); curves {c['curves']} / {n['curves']}; wired both ends {c['wired_both_ends']} / {n['wired_both_ends']}; " + ", ".join(f"{k} {hc[k]:.4f} -> {hn[k]:.4f} ({d[k]:+.4f})" for k in K))
    print(f"        bootstrap n - c topo F1 {iv[0]:+.4f} [{iv[1]:+.4f}, {iv[2]:+.4f}]")
    assert n["long_links_dropped"] == 0, "the patch did not switch the cover drop off"
    reads.append((d, iv))
dt = [r[0]["topo_F1"] for r in reads]
dd = [r[0]["dup_frac"] for r in reads]
if not fx_all:
    r4 = "INVALID (a fixture failed or is missing)"
elif all(abs(x) < 0.003 for x in dt):
    r4 = "FREE (|d topo F1| < 0.003 on both seeds)"
elif all(x <= -0.003 for x in dt) and all(r[1][2] < 0 for r in reads):
    r4 = f"CARRIES {min(-x for x in dt):.4f} topo F1 (the smaller |d| of the two seeds; both d <= -0.003, bootstrap upper bounds below 0)"
elif all(x >= 0.003 for x in dt) and all(r[1][1] > 0 for r in reads):
    r4 = "HARMS (switching the cover drop off is better by >= 0.003 on both seeds, lower bounds above 0)"
else:
    r4 = "MIXED"
print("R4:", r4, "| dup_frac differences", [f"{x:+.4f}" for x in dd], "| prediction d -0.008 (between -0.02 and -0.003):", [f"{x:+.4f}" for x in dt])
m = json.load(open(f"{ROOT}/temp/e364/mask_ceiling.json"))
print(f"R6 (a): {m['admitted']} of {m['pairs']} true connectors admitted = {m['admitted_share']:.4f} (frames {m['frames']}, with pairs {m['frames_with_pairs']}); failing turn {m['fail_turn']}, chord < {m['min_chord']} m {m['fail_chord']}, forward a {m['fail_fwd_a']}, forward b {m['fail_fwd_b']}: {m['reading']}; prediction 0.98")
for s in ("0", "1"):
    r = m["model_reality"][s]
    print(f"R6 (b) seed {s}: both ends matched {r['both_ends_matched']} of {r['pairs']} = {r['both_ends_matched'] / r['pairs']:.3f}; reachable (stored allowed True) {r['reachable']} = {r['reachable_of_matched']:.4f} of the matched, {r['reachable_of_all']:.4f} of all true connectors")
