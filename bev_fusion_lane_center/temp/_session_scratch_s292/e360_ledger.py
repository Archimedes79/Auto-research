"""Ledger notes of the nine E360 scoring jobs; numbers are read from the stored cards (not typed). usage (problem folder, WSL): python3 temp/_session_scratch_s292/e360_ledger.py"""
import json, subprocess
R = "runs/e324_s_s%d/"
K = ("topo_F1", "dup_frac", "F1@1.0", "junction_F1", "lanes_found", "vert100_dp")
P = "python3 ../autoresearch/tools/ledger.py note".split()
def card(s, tag, suf):
    return json.load(open(R % s + f"ends_card_{tag}_{suf}_thr0.1.json"))
def fmt(c):
    h = c["numbers"]["head"]
    return ", ".join(f"{k} {h[k]:.4f}" for k in K) + f"; curves {c['curves']}, wired both ends {c['wired_both_ends']}"
def note(i, result, decision):
    r = subprocess.run(P + [i, "--result", result, "--decision", decision], capture_output=True, text=True)
    print(i, r.returncode, (r.stdout + r.stderr).strip()[:100])
base = "by hand: D87 replay (temp/e360/verdict_e360.py), card E360 section 6"
t = card(0, "e360t", "snap_nogeom_attach"); e = card(0, "e338a", "snap_nogeom_attach")
same = all(t[k] == e[k] for k in ("numbers", "curves", "wired_both_ends", "long_links_dropped", "ends_within_1m_of_a_delivered_piece_end"))
note("E360.t_s0", f"c2 graph WITH the hard step at PRIMARY: {fmt(t)}; equals the stored E338a card in numbers, curves, wired, long links, ends_within_1m: {same} (counts file identical, checked)", "fixture passed (stop rule (v) held); " + base)
for s in (0, 1):
    c, a, ck, ak = card(s, "e360c", "snap_nogeom_attach"), card(s, "e360a", "snap_nogeom_attach"), card(s, "e360ck", "snap_nogeom"), card(s, "e360ak", "snap_nogeom")
    note(f"E360.c_s{s}", f"end-to-end control (c2 plain, no hard step, no layer), PRIMARY read-out: {fmt(c)}", "control row, no token; " + base)
    note(f"E360.a_s{s}", f"layer arm (E292 a, no hard step), PRIMARY: {fmt(a)}; a - c topo F1 {a['numbers']['head']['topo_F1'] - c['numbers']['head']['topo_F1']:+.4f}, dup_frac {a['numbers']['head']['dup_frac'] - c['numbers']['head']['dup_frac']:+.4f} (guard 0.01 BROKEN), junction F1 {a['numbers']['head']['junction_F1'] - c['numbers']['head']['junction_F1']:+.4f}",
         "INVALID by the pre-registered stop rule (iii) (attached - keypoint share within 1 m, tolerance 0.01 written before the results: a entry/exit " + ", ".join(f"{a['ends_within_1m_of_a_delivered_piece_end'][k] - a['ends_within_1m_of_a_delivered_piece_end']['keypoint_' + k]:+.4f}" for k in ('entry', 'exit')) + " vs c " + ", ".join(f"{c['ends_within_1m_of_a_delivered_piece_end'][k] - c['ends_within_1m_of_a_delivered_piece_end']['keypoint_' + k]:+.4f}" for k in ('entry', 'exit')) + "); nothing cut, nothing adopted; with the first tolerance 0.02 the token would be TIE (dup_frac guard broken); " + base)
    note(f"E360.ck_s{s}", f"witness control (own keypoints, snap-nogeom, c2 plain): {fmt(ck)}", "witness row, no token; " + base)
    note(f"E360.ak_s{s}", f"witness layer (own keypoints, snap-nogeom, a): {fmt(ak)}; ak - ck topo F1 {ak['numbers']['head']['topo_F1'] - ck['numbers']['head']['topo_F1']:+.4f}", "witness row: same sign and size as PRIMARY; " + base)
