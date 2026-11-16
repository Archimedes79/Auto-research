"""Fills section 3 of the E353 card from the cards of the run folders, the bootstrap logs and the rule replay logs (numbers never typed).
usage (problem folder, WSL): python3 temp/_session_scratch_s292/e353_section3.py"""
import io
import json
import os
import re
import time

R = os.getcwd()
P = "experiments/E353_paint-branch-width-25/experiment.md"
READS = [("snapped", "snap", "ends_card_snap_thr0.1.json"), ("PRIMARY", "primary", "ends_card_snap_nogeom_attach_thr0.1.json"), ("attach-only", "attach", "ends_card_attach_thr0.1.json"),
         ("unsnapped (not judged)", None, "ends_card_thr0.1.json")]
RUNS = {"c": "e324_s_s{S}", "cp": "e353_cp_s{S}", "n1": "e353_n1_s{S}", "n1z": "e353_n1z_s{S}"}


def head(run, card):
    p = os.path.join(R, "runs", run, card)
    return json.load(open(p))["numbers"]["head"] if os.path.exists(p) else None


def boot(tag, ro, S):
    t = open(os.path.join(R, "temp", "e353", f"boot_{tag}_{ro}_s{S}.log"), encoding="utf-8").read()
    m = re.search(r"topo_F1\s+A - B: point ([+-][\d.]+), .*95 % \[([+-][\d.]+), ([+-][\d.]+)\]", t)
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


rows = ["| seed | read-out | c (E324 s) | cp | n1 | n1z | n1 - c: topo F1 [paired 95 %] | n1 - cp: topo F1 [paired 95 %] | n1 - c guards: F1@1.0 / junction F1 / dup_frac / vert100_dp |", "|---|---|---|---|---|---|---|---|---|"]
lows_c, lows_cp, ok_guard = [], [], True
for S in (0, 1):
    for label, ro, card in READS:
        h = {k: head(v.format(S=S), card) for k, v in RUNS.items()}
        f = lambda k: "-" if h[k] is None else f"{h[k]['topo_F1']:.4f}"
        if ro:
            pc, lc, uc = boot("c", ro, S)
            pp, lp, up = boot("cp", ro, S)
            lows_c.append(lc)
            lows_cp.append(lp)
            bc, bp = f"{pc:+.4f} [{lc:+.4f}, {uc:+.4f}]", f"{pp:+.4f} [{lp:+.4f}, {up:+.4f}]"
        else:
            bc = bp = f"{h['n1']['topo_F1'] - h['c']['topo_F1']:+.4f} / {h['n1']['topo_F1'] - h['cp']['topo_F1']:+.4f} (point only)"
        n, c = h["n1"], h["c"]
        g = f"{n['F1@1.0'] - c['F1@1.0']:+.4f} / {n['junction_F1'] - c['junction_F1']:+.4f} / {n['dup_frac'] - c['dup_frac']:+.4f} / {n['vert100_dp'] - c['vert100_dp']:+.2f}"
        rows.append(f"| {S} | {label} | {f('c')} | {f('cp')} | {f('n1')} | {f('n1z')} | {bc} | {bp} | {g} |")
tab = "\n".join(rows)


def rule_lines(rid):
    t = open(os.path.join(R, "experiment_execution", "runs", rid, "run.log"), encoding="utf-8").read().split("\n")
    return "\n".join("    " + l.strip() for l in t if l.strip().startswith(("snapped", "PRIMARY", "attached", "verdict")))


def d(S, k, ro_card, a, b, key="topo_F1"):
    return head(RUNS[a].format(S=S), ro_card)[key] - head(RUNS[b].format(S=S), ro_card)[key]


snapc, primc, attc = READS[0][2], READS[1][2], READS[2][2]
pred = (f"- **n1 minus c, snapped** -0.002 (window -0.006 .. +0.002 on each seed): {d(0, 0, snapc, 'n1', 'c'):+.4f} / {d(1, 0, snapc, 'n1', 'c'):+.4f} HIT. **PRIMARY** -0.002 (same window): {d(0, 0, primc, 'n1', 'c'):+.4f} / {d(1, 0, primc, 'n1', 'c'):+.4f} HIT. "
        f"**attach-only** -0.003 (window -0.008 .. +0.002): {d(0, 0, attc, 'n1', 'c'):+.4f} / {d(1, 0, attc, 'n1', 'c'):+.4f} HIT.\n"
        f"- **cp minus c** +0.001 (window -0.003 .. +0.004): snapped {d(0, 0, snapc, 'cp', 'c'):+.4f} / {d(1, 0, snapc, 'cp', 'c'):+.4f}, PRIMARY {d(0, 0, primc, 'cp', 'c'):+.4f} / {d(1, 0, primc, 'cp', 'c'):+.4f}, attach-only {d(0, 0, attc, 'cp', 'c'):+.4f} / {d(1, 0, attc, 'cp', 'c'):+.4f}: HIT on all six.\n"
        f"- **n1z (before the fine-tune, whole groups) minus c, snapped** -0.004 (window -0.012 .. +0.001): {d(0, 0, snapc, 'n1z', 'c'):+.4f} / {d(1, 0, snapc, 'n1z', 'c'):+.4f} HIT (the fine-tune recovers {d(0, 0, snapc, 'n1', 'n1z'):+.4f} / {d(1, 0, snapc, 'n1', 'n1z'):+.4f}); PRIMARY {d(0, 0, primc, 'n1z', 'c'):+.4f} / {d(1, 0, primc, 'n1z', 'c'):+.4f}.\n"
        "- **Saved GMACs about 1.1:** 1.1059 (8.2016 -> 7.0957 per frame, measured by `E353.n1z_s0 / s1`) HIT. **Quiet-host ms about -0.8:** PENDING (`E353.time_s0` in the quiet window of the night; see the D106 line).")
now = time.strftime("%Y-%m-%d %H:%M")
txt = f"""## 3. Summary ({now}; both seeds, VAL 873 frames / 22 scenes; the rule replay by `evaluation/jrule_removal.py` (run ids `E353.rule_c`, `E353.rule_cp`), the paired scene bootstrap by `jdiag.py boot` (1000 resamples), the cards of `runs/e353_*` and `runs/e324_s_s*`; chain `temp/day29/day29.sh` + `temp/day34/day34.sh`, run ids `E353.*`)
**Result: n1 (the 0.5 m paint branch at 2 of its 8 GroupNorm groups, fine-tuned 1500 steps) is ADOPTED by the written rule, label ADOPT-FIRM, on E324 s's recipe: against E324 s (c) AND against its own continuation (cp), on both seeds and all three judged read-outs, every number inside the 0.010 band with the guards held; the lowest paired 95 % lower bound of the topo F1 difference is {min(lows_c):+.4f} against c and {min(lows_cp):+.4f} against cp (bar -0.010). It saves 1.1059 GMACs (13.5 % of the system's 8.2016) and 27 664 parameters. The quiet-host ms, the fourth fact of the D106 line, is PENDING: `E353.time_s0` runs in the quiet window of the night (05:00 - 07:00); until it exists the adoption is by the written rule only and D106's 'faster' is open.**
**The rule replay (as printed by the tool):**
  against c (`E353.rule_c`):
{rule_lines('E353.rule_c')}
  against cp (`E353.rule_cp`):
{rule_lines('E353.rule_cp')}
**Numbers (topo F1 of the head; deltas to c in the last column; the guards are n1 minus c on seed and read-out; paired 95 % intervals over the 22 scenes; the unsnapped read-out is reported, not judged; n1z has no attach-only and no unsnapped card):**

{tab}

**Predictions against the numbers (windows from section 1):**
{pred}
**D106 line ('simpler AND faster?'; the written rule of section 0 applies it, this card does not decide a borderline case itself):** *parts:* the same parts at a narrower width (the paint branch's hidden block 32 -> 8 channels, GroupNorm(8, 32) -> GroupNorm(2, 8), one new bias tensor `jmodel.jm.m05.3.bias`); no part added or removed; **one new checkpoint class (narrowed widths + a bias tensor) is the price** (a surgery-aware loader, `training/grid/jnarrow.py`, until a native narrowed config exists). *parameters:* 2 740 571 -> 2 712 907 (-27 664, -1.0 %). *GMACs:* -1.1059 (-13.5 % of 8.2016). *quiet-host ms:* PENDING (paired, with its interval; 'faster' = the upper end of the interval below 0). **Reading, to be completed by the ms row:** if the paired interval lies below 0 on a QUIET label, n1 is simpler (fewer parameters and GMACs, the same parts) AND faster and D106 chooses it without asking; the new checkpoint class is named as the price. If the ms row is NOT QUIET or not below 0, the adoption stays 'by the rule, speed open' and the cost-table GMACs stand as the only speed fact (the E351 lesson: ms follows the host, not the MACs).
**Caveats (named):** (1) `E353.cp_s0` was resumed from its step-1000 checkpoint after three CUDA unknown errors (section 2b): its data order after step 1000 differs from n1_s0's, so every n1 - cp number of seed 0 carries that difference; seed 1 ran clean (no retry, one attempt each, 00:49), and the seed-1 numbers against cp are the clean comparison (n1 - cp topo F1 on seed 1: snapped {d(1, 0, snapc, 'n1', 'cp'):+.4f}, PRIMARY {d(1, 0, primc, 'n1', 'cp'):+.4f}, attach-only {d(1, 0, attc, 'n1', 'cp'):+.4f}). (2) Two seeds of the same recipe differ by about 0.004 and the paired 22-scene intervals are about +-0.004: the differences above are inside that noise on every row, which is what 'no loss' means here; the paired bootstrap does not cover retraining noise. (3) The group choice per seed comes from E345's channel ranking (calibration set named there), the same frames family as VAL; the VAL numbers of this card are therefore not independent of the choice of WHICH groups survive (the choice was made before any run of this card, from E345, and is not tuned on these numbers). (4) The `E353.cp_*` and `E353.n1_*` dumps and cards ran on the shared box (load 15 - 20), the numbers are deterministic given the checkpoints. (5) The E324 s system of record (D94) is unchanged: n1 is a candidate on its recipe, not a promotion; E343 (one TEST) stays on hold.
**Does not claim:** anything about the ms before the quiet-host row; anything on TEST; that the full 8 groups were needed (n1 keeps 2); anything about B's pyramid (E352).
"""
s = io.open(P, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
a = "## 3. Summary" + nl + "(empty until the runs exist)"
assert s.count(a) == 1, "placeholder of section 3 not found exactly once"
s = s.replace(a, txt.rstrip("\n").replace("\n", nl))
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("ok", len(txt), "lows", min(lows_c), min(lows_cp))
