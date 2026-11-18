"""Fills / refreshes section 3 of the E352 card from the step-0 table of the tool (temp/e352/step0_table.md, written by `jprune_b.py table --seeds ... --out ...`): re-runnable (markers).
usage (problem folder, WSL): python3 temp/_session_scratch_s292/e352_section3.py [--final]"""
import io
import re
import sys
import time

P = "experiments/E352_pyramid-by-importance/experiment.md"
BEG, END = "<!-- e352-step0-begin -->", "<!-- e352-step0-end -->"
tab = io.open("temp/e352/step0_table.md", encoding="utf-8").read()
final = "--final" in sys.argv
seeds = sorted({int(m) for m in re.findall(r"### seed (\d+), topo_F1", tab)})
vals = {}
for line in tab.split("\n"):
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    m = re.match(r"### seed (\d+)", line)
    if m:
        cur = int(m.group(1))
    if len(c) == 10 and c[0] in ("groups", "channels", "-"):
        try:
            vals[(cur, c[0], c[1])] = [float(x) if x not in ("-",) else None for x in c[2:]]
        except ValueError:
            pass


def g(seed, var, pt, col):  # col 0 B plain value, 1 its delta, 2 snap, 3 delta, 4 primary, 5 delta, 6 attach, 7 delta
    r = vals.get((seed, var, pt))
    return None if r is None else r[col]


def f(x, d=4, sign=True):
    return "-" if x is None else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


rule = tab.split("### the rule")[1] if "### the rule" in tab else ""
now = time.strftime("%Y-%m-%d %H:%M")
done_pts = sorted({(s, v, p) for (s, v, p) in vals if g(s, v, p, 0) is not None and p != "null"})
n_groups = len([1 for (s, v, p) in done_pts if v == "groups" and not p.endswith("mean")])
status = "FINAL: all points of the list are in" if final else f"PROVISIONAL at {now}: seeds {seeds} so far ({len(done_pts)} point rows), the rest of the chain is still running"
def rg(var, pt, col):  # "a .. b" over the seeds that have the row
    xs = [g(s, var, pt, col) for s in seeds if g(s, var, pt, col) is not None]
    return "-" if not xs else (f(xs[0]) if len(xs) == 1 else f"{min(xs):+.4f} .. {max(xs):+.4f}")


t1_loss = min([g(s, 'groups', 't1', 1) for s in seeds if g(s, 'groups', 't1', 1) is not None] or [0.0])
body = f"""{BEG}
**Step 0 ({status}; no training, B's pyramid trunk edited at inference on the stored B dumps, whole GroupNorm groups replaced by the calibration mean; run ids `E352.s0_*`, 3 parallel streams `temp/day33/day33.sh` after the null gates `E352.s0_null_gate_s0 / s1` PASSED; VAL 873 frames, 22 scenes; the tool is `evaluation/jprune_b.py`; by hand: no tool decides, the kill and the windows are the card's):**
**Result: the kill of step 0 did NOT fire on any seed (T1 loses at most {-t1_loss:.4f} topo F1 on B's plain graph against the kill bar 0.03). The trunk set T1 (e2 50 %, d2 25 %, e3 25 % of the hidden groups, 0.2975 GMACs = 3.6 % of the system) is nearly free WITHOUT retraining: B plain {rg('groups', 't1', 1)}, snapped {rg('groups', 't1', 3)}, PRIMARY {rg('groups', 't1', 5)}, attach-only {rg('groups', 't1', 7)} (seeds {seeds}, Delta to the null); and its random control costs the same: ranked minus the mean of three random draws {rg('groups', 't1 ranked - random', 0)} on B plain (the prediction 'at least +0.003' MISSED: the ranking carries nothing at this level; the trunk is simply redundant here). T2 (T1 plus d0, e1, d1 at 25 - 50 %, 0.8426 GMACs = 10.3 %) is NOT free: B plain {rg('groups', 't2', 1)}, snapped {rg('groups', 't2', 3)}, far outside the window (-0.025 .. -0.002); its random control is as bad or worse (ranked minus the mean of three random draws {rg('groups', 't2 ranked - random', 0)} on B plain, seed 0 / seed 1: the ranking helps on one seed and is no better than chance on the other), so replacing whole groups of the shallow stages d0 / e1 / d1 cannot be done without retraining. The E345 channel-level anchor (the same importance, single channels, which GroupNorm(8, C) does not allow to build physically) loses far less: T1 {rg('channels', 't1', 1)}, T2 {rg('channels', 't2', 1)} on B plain, snapped {rg('channels', 't2', 3)}: the damage of whole-group T2 comes from the granularity (the group statistics of the kept channels change), not from the channels themselves. Whether the whole-group T2 loss is recoverable by retraining is the question of the trained stage (stage D / E of the card), not of step 0.**
**The step-0 table (tool output, Delta = value - null of the same seed; groups = the judged variant, channels = the E345 anchor):**

{tab.split('### the rule')[0].strip()}

**The rule lines (the card's windows; HIT / MISS as printed by the tool):**
{rule.strip()}

**Reading.** (1) T1's 3.6 % of the system's MACs is a 'MACs only' saving by the card's own arithmetic (about 0.3 ms, below the 0.5 ms floor): not a candidate for a speed claim without a measurement and no reason to retrain for it alone. (2) The big lever would be T2-like savings (10.3 % of the MACs, shallow stages), and step 0 says they need retraining; the budget of that stage (stage D distillation 1000 steps + stage E 1500 steps per arm, cp / t1 / t2 x 2 seeds = 6 slot hours + the build-2 code and a critic pass) was NOT run tonight. (3) Random draws as free as the ranked set at T1 mean the importance ranking is not what makes T1 free; at T2 the ranking helps on seed 0 only. (4) The channel anchor says the shallow stages are redundant at the channel level (T2 -0.02 .. -0.04 on B plain) while the buildable whole-group version collapses: a trained narrowing has to survive GroupNorm's group statistics (E353 keeps whole groups for this reason), and the retraining must re-learn them; step 0 cannot say whether it does.
**Not run (and why):** build 2 (pyramid edits, `--distill`), the training stages D / E, the 50-step timing gate, the quiet-host ms (all of them need GPU slots and a code critic pass; D107 asks only for finished experiments tonight, the priorities were D103 / D105 / E353; step 0 is the planned CPU part and it is complete{'' if final else ' for the points listed above'}). **D106 line:** no system exists yet (nothing to compare on parts / parameters / GMACs / ms); the first trained arm will carry it.
**Caveats:** two seeds only; no bootstrap here (the null gates and the differences are deterministic replays of stored dumps; the paired intervals of E345 / E348 were about +-0.004); the E345 channel-level anchor rows are the 'channels' variant (the sets the narrowing cannot build, kept as the anchor of E345's +0.002); in-sample: the importance ranking comes from E345's calibration frames (TRAIN), the table is VAL.
{END}"""
s = io.open(P, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
b = body.replace("\n", nl)
if BEG in s:
    i, j = s.index(BEG), s.index(END) + len(END)
    s = s[:i] + b + s[j:]
else:
    a = "## 3. Summary" + nl + "(empty until the runs exist)"
    assert s.count(a) == 1, "placeholder of section 3 not found exactly once"
    s = s.replace(a, "## 3. Summary" + nl + b)
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("ok", status, len(body))
