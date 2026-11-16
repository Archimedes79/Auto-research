"""Ledger notes of the E352 step-0 point jobs, numbers parsed from temp/e352/step0_table.md (the tool's table), not typed.
usage (problem folder, WSL): python3 temp/_session_scratch_s292/e352_ledger.py"""
import io
import re
import subprocess

tab = io.open("temp/e352/step0_table.md", encoding="utf-8").read()
vals = {}
cur = None
for line in tab.split("\n"):
    m = re.match(r"### seed (\d+)", line)
    if m:
        cur = int(m.group(1))
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(c) == 10 and c[0] in ("groups", "channels", "-") and cur is not None:
        try:
            vals[(cur, c[0], c[1])] = [float(x) if x != "-" else None for x in c[2:]]
        except ValueError:
            pass
P = "python3 ../autoresearch/tools/ledger.py note".split()


def note(rid, result, decision):
    r = subprocess.run(P + [rid, "--result", result, "--decision", decision], capture_output=True, text=True)
    print(rid, r.returncode, (r.stdout + r.stderr).strip()[:90])


def fmt(S, var, pt):
    r = vals.get((S, var, pt))
    if r is None:
        return None
    return f"B plain {r[1]:+.4f} (null {vals[(S, '-', 'null')][0]:.4f}), snapped {r[3]:+.4f}, PRIMARY {r[5]:+.4f}, attach-only {r[7]:+.4f}" if r[5] is not None else f"B plain {r[1]:+.4f}, snapped {r[3]:+.4f}"


for S in (0, 1):
    n = vals[(S, "-", "null")]
    note(f"E352.s0_null_gate_s{S}", f"null gate seed {S} PASSED (the unedited B' dump through the wrapper equals the stored dump within the rule's tolerance, three read-outs); null topo F1: B plain {n[0]:.4f}, snapped {n[2]:.4f}, PRIMARY {n[4]:.4f}, attach-only {n[6]:.4f}", "gate passed (by hand: replay), the chain went on")
    for pt, label in (("t1", "T1 whole groups (e2 50 %, d2 25 %, e3 25 %; 0.2975 GMACs = 3.6 %)"), ("t2", "T2 whole groups (T1 + d0, e1, d1; 0.8426 GMACs = 10.3 %)")):
        k = "g"
        s = fmt(S, "groups", pt)
        if s:
            dec = "kill not fired (T1 loses <= 0.0030 on B plain, bar 0.03); T1 nearly free, ranking = random draws (prediction at least +0.003 MISSED)" if pt == "t1" else "T2 NOT free without retraining (window -0.025 .. -0.002 MISSED)"
            note(f"E352.s0g_{pt}_s{S}", f"{label}, seed {S}: {s}", dec + " (by hand: replay, jprune_b.py table)")
        s = fmt(S, "channels", pt)
        if s:
            note(f"E352.s0c_{pt}_s{S}", f"E345 channel-level anchor {pt.upper()}, seed {S}: {s}", "anchor row (not judged): the channel-level damage is far smaller than the whole-group damage at T2 (by hand: replay)")
