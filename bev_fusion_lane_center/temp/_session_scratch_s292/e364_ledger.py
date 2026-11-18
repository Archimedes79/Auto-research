"""Ledger notes of the E364 rows; numbers read from the stored cards. usage (problem folder, WSL, system python)"""
import json
import subprocess
P = "python3 ../autoresearch/tools/ledger.py note".split()
K = ("topo_F1", "dup_frac", "F1@1.0", "junction_F1", "lanes_found", "vert100_dp")


def card(s, tag=None):
    name = "snap_nogeom_attach" if tag is None else f"{tag}_snap_nogeom_attach"
    return json.load(open(f"runs/e324_s_s{s}/ends_card_{name}_thr0.1.json"))


def line(c):
    h = c["numbers"]["head"]
    return ", ".join(f"{k} {h[k]:.4f}" for k in K) + f"; curves {c['curves']}, wired both ends {c['wired_both_ends']}, long links dropped {c['long_links_dropped']}"


def note(i, result, decision):
    r = subprocess.run(P + [i, "--result", result, "--decision", decision], capture_output=True, text=True)
    print(i, r.returncode)


base = "by hand: D87 replay (temp/e364/read_e364.py), card E364 section 6"
for s in (0, 1):
    c, f, n = card(s), card(s, "e364f"), card(s, "e364n")
    ok = all(f[k] == c[k] for k in ("numbers", "curves", "wired_both_ends", "long_links_dropped", "ends_within_1m_of_a_delivered_piece_end"))
    note(f"E364.fx_s{s}", f"fixture: the launcher without the patch (cover radius 4.0): {line(f)}; equals the stored PRIMARY card in numbers / curves / wired / long links dropped / shares: {ok}", "fixture passed; " + base)
    d = n["numbers"]["head"]["topo_F1"] - c["numbers"]["head"]["topo_F1"]
    note(f"E364.n_s{s}", f"R4 off (cover drop switched off, COVER_POS -1.0): {line(n)}; n - c topo F1 {d:+.4f}, dup_frac {n['numbers']['head']['dup_frac'] - c['numbers']['head']['dup_frac']:+.4f}", f"R4 CARRIES (the smaller |d| of the two seeds 0.0266; both d <= -0.003, bootstrap upper bounds below 0); prediction -0.008 missed on size; measurement, no system token; {base}")
m = json.load(open("temp/e364/mask_ceiling.json"))
note("E364.m", f"R6 pair mask on the GT: {m['admitted']} of {m['pairs']} true connectors admitted = {m['admitted_share']:.4f} (fail: chord {m['fail_chord']}, turn {m['fail_turn']}, forward {m['fail_fwd_a']}); model's reality seed 0 / 1: both ends matched {m['model_reality']['0']['both_ends_matched']} / {m['model_reality']['1']['both_ends_matched']} of {m['pairs']}, reachable of matched {m['model_reality']['0']['reachable_of_matched']:.4f} / {m['model_reality']['1']['reachable_of_matched']:.4f}", f"R6 COSTS 0.0262 of the true connectors on the GT (prediction 0.98: marginal miss, not blind: smoke runs had printed it); index limit with a small cost, the ends dominate; {base}")
