"""Ledger notes of the E358 rows; numbers read from the stored cards. usage (problem folder, WSL, system python): python3 temp/_session_scratch_s292/e358_ledger.py"""
import json, subprocess
P = "python3 ../autoresearch/tools/ledger.py note".split()
K = ("topo_F1", "dup_frac", "F1@1.0", "junction_F1", "lanes_found", "vert100_dp")
def card(s, tag, suf="snap_nogeom_attach"):
    return json.load(open(f"runs/e324_s_s{s}/ends_card_{tag}_{suf}_thr0.1.json"))
def line(c):
    h = c["numbers"]["head"]
    return ", ".join(f"{k} {h[k]:.4f}" for k in K)
def note(i, result, decision):
    r = subprocess.run(P + [i, "--result", result, "--decision", decision], capture_output=True, text=True)
    print(i, r.returncode)
base = "by hand: D87 replay (jwire_verdict.py --read-out primary), card E358 section 6"
for s in (0, 1):
    c = card(s, "e358w"); k = json.load(open(f"runs/e324_s_s{s}/ends_card_snap_nogeom_attach_thr0.1.json"))
    d = c["numbers"]["head"]["topo_F1"] - k["numbers"]["head"]["topo_F1"]
    wa = json.load(open(f"runs/e358_w_s{s}/wire_agreement.json"))
    note(f"E358.w_s{s}", f"head on the ATTACHED ends of E324 s, training run seed {s}: heads 0 - 2 + distance-only + no-jitter written; VAL real-choice agreement head 0 vs rule: see wire_agreement.json (runs/e358_w_s{s}); kill not fired; 545 parameters", "training row, token on the scoring row: " + base)
    note(f"E358.sc_s{s}", f"end-to-end read-out with head 0: {line(c)}; control E324 s PRIMARY: {line(k)}; w - c topo F1 {d:+.4f}", f"KEEP+ (interval above 0 on both seeds, topo F1 >= c - 0.005 and all six guards held, kill not fired, prediction +0.001 HIT within 0.003); {base}")
