import re
p = "experiments/E241_early-kill-learning-curves/experiment.md"
import glob
p = glob.glob("experiments/E241*/experiment.md")[0]
t = open(p, encoding="utf-8").read()
i = t.index("## 3. Summary")
head, tail = t[:i], t[i:]
new = """## 3. Summary (2026-10-26 / 27, run E241.ev2 plus the pair-3 evaluations of loop 20; VAL 873, `evaluate.py`; scripts `temp/loop19/e241_prep.py`, `eval_e241.sh`, `temp/loop20/e241_cross.py`)
- **Decision by the written rule (by hand, the tool has no cross-pairing rule): KEEP in the sense of the card, "worth a larger replay", not "switch the tick on".** Cut test at 50 % on the four cross-pairings of seeds 2 / 3 (j1rc against j1eq, lanes found, VAL 873): j1rc is above j1eq by +0.0755 / +0.0738 / +0.0534 / +0.0516 (correct cut for j1eq fires in 4 of 4, threshold 0.02; wrong cut for j1rc in 0 of 4). The bar was 0 of 4 wrong and 3 of 4 correct: passed. As the card says, 0 of 4 non-independent pairings bounds the wrong-cut rate only to about 0.5 (95 % upper bound), so the tick may cut on lanes found only after 12 pairings; the production rule is not adopted.
- **Lanes found, j1rc - j1eq per cross-pairing (rc2-eq2 / rc2-eq3 / rc3-eq2 / rc3-eq3):** 25 %: +0.078 / +0.077 / +0.066 / +0.065; 50 %: +0.076 / +0.074 / +0.053 / +0.052; 75 %: +0.046 / +0.054 / +0.048 / +0.056; final: +0.053 / +0.054 / +0.056 / +0.058. The effect has its full size at 25 % and is larger there than at the end (+0.066 - 0.078 against +0.053 - 0.058): an early reading overstates the final gap by up to a third.
- **Q, same order:** 25 %: +0.008 / +0.011 / +0.006 / +0.009; 50 %: +0.007 / +0.012 / +0.005 / +0.011; 75 %: +0.001 / +0.006 / +0.003 / +0.009; final: +0.002 / +0.004 / +0.006 / +0.007. Q does not separate noise from effect at any checkpoint (paired SD 0.0043 from E240; the pairings share runs): do not cut or promote on Q.
- **Prediction (card):** at 50 % lanes found +0.045 (0.02 - 0.065): measured +0.052 to +0.076 (mean +0.063): two pairings above the range, sign and cut test right; Q at 50 % inside +-0.010: two of four pairings at +0.0105 / +0.0118 (just outside). At 25 % the card expected a pairing below +0.02: missed (all four at least +0.065), so 25 % is NOT too early for this effect, only for this type of effect.
- **Scope, referee points kept:** loss-weight type effects only (the recall weight changes lanes found from the first steps, so the test is nearly built to pass); it says nothing about architecture or regularisation changes; intermediate checkpoints are not annealed (OneCycle, 59 % of peak at 50 %). Pair 3 was trained on the stored distance map, pair 2 as well (loop 19), so the cross-pairings share one map. At 50 % and 75 % j1rc has +0.4 to +0.6 more vertices per 100 m than j1eq (pair 3: +0.4 at 50 %, +0.0 at 75 %), which inflates both gaps a little.
- **System finding:** `decide.py --experiment` reads one card per arm and has no cross-pairing cut test, so E241 is decided by hand from a throw-away table (`temp/loop20/e241_cross.py`); a reusable cut-test subcommand is worth building when the 12-pairing replay is run (N15).
- **Next:** the 12-pairing replay would need seeds 4 - 5 (0.9 GPU h); queued as a possible piggyback on the next recall-type fine-tunes, not run now.
"""
open(p, "w", encoding="utf-8").write(head + new)
