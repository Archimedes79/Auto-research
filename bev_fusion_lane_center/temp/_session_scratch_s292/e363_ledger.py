"""Ledger notes of the E363 rows; numbers read from the stored cards. usage (problem folder, WSL, system python): python3 temp/_session_scratch_s292/e363_ledger.py"""
import json
import subprocess
P = "python3 ../autoresearch/tools/ledger.py note".split()
K = ("topo_F1", "dup_frac", "F1@1.0", "junction_F1", "lanes_found", "vert100_dp")


def card(s, g, tag):
    return json.load(open(f"runs/e363_{g}_s{s}/ends_card_{tag}_snap_nogeom_attach_thr0.1.json"))


def line(c):
    h = c["numbers"]["head"]
    return ", ".join(f"{k} {h[k]:.4f}" for k in K) + f"; curves {c['curves']}, wired both ends {c['wired_both_ends']}"


def note(i, result, decision):
    r = subprocess.run(P + [i, "--result", result, "--decision", decision], capture_output=True, text=True)
    print(i, r.returncode)


base = "by hand: D87 replay (temp/e363/verdict_e363.py), card E363 section 6"
for s in (0, 1):
    c, t, a, cw, aw = card(s, "c2", "e363c"), card(s, "c2", "e363t"), card(s, "a", "e363a"), card(s, "c2", "e363cw"), card(s, "a", "e363aw")
    f = lambda x: x["numbers"]["head"]["topo_F1"]  # noqa: E731
    note(f"E363.c_s{s}", f"faithful end-to-end control (c2 plain, attachment on c2's own node head, no layer, no hard step): {line(c)}", "control row, no token; " + base)
    note(f"E363.t_s{s}", f"faithful rule-version arm (c2 WITH the hard close-node step): {line(t)}; c - t {f(c) - f(t):+.4f}", "reference row, no token; " + base)
    note(f"E363.a_s{s}", f"layer arm (E292 a, Matrix-NMS, no hard step, attachment on a's node head AFTER the layer): {line(a)}; a - c topo F1 {f(a) - f(c):+.4f}, dup_frac {a['numbers']['head']['dup_frac'] - c['numbers']['head']['dup_frac']:+.4f} (bound 0.010 BROKEN), a - t {f(a) - f(t):+.4f}",
         "TIE, kept and refined (D87): every stop rule (i) - (vii) held, topo F1 and five guards held, bootstrap lower bound above 0, witness condition (b) within 0.01, duplicate guard broken on both seeds; nothing adopted (D110 open); " + base)
    note(f"E363.cw_s{s}", f"E358 head on the control graph: {line(cw)}; cw - c {f(cw) - f(c):+.4f}", "stack control, no token; " + base)
    note(f"E363.aw_s{s}", f"THE STACK (layer + E358 head, no hard step, no distance rule): {line(aw)}; aw - a {f(aw) - f(a):+.4f}, aw - c {f(aw) - f(c):+.4f}, against E324 s PRIMARY on B with both rule steps {f(aw) - (0.6912 if s == 0 else 0.6948):+.4f}", "stack row: head transfer HIT (window and bootstrap above 0); the stack is reported as a candidate, nothing adopted; " + base)
for s in (0, 1):
    for g in ("c2", "a", "b"):
        i = f"E363.dump_{g}_s{s}"
        if g == "b" and False:
            continue
        note(i, f"faithful dump of the pair run with the node head of {'B itself (fixture)' if g == 'b' else 'B-prime ' + g} (873 frames)" if g != "a" else "FIRST dump of arm a WITHOUT the Matrix-NMS layer in front of Attach (replaced, moved to ends_val_nolayer.pkl, never scored)", "data preparation (--no-card); acceptance = the stop rules of the card E363 (surgery_check.py arrays, fixture bit for bit on both seeds, node check): " + ("PASS" if g != "a" else "superseded by E363.dumpL_a_s<S>"))
    note(f"E363.dumpL_a_s{s}", "layer-faithful dump of arm a (the Matrix-NMS decay of B' applied to the node outputs before Attach by the in-process wrapper temp/e363/faithful.py), 873 frames", "data preparation (--no-card); acceptance = surgery_check.py PASS and the node check (e, pos, e_raw equal QGraph.node_stage of B' to 0.0 on 16 frames, seeds 0 and 1)")
note("E358.sc_s0", "(addendum 14:50) the staged scorer of this row had the derived-run rule added to `check_head` AFTER this scoring (E363); the hunk is in staged_diff.patch and E358 did not run with it", "bookkeeping, no token change")
