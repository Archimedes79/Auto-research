import io
import subprocess
import time

p = "experiments/E291_sd-road-level-lane-counts/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
assert "## 5. Result" not in s
tab = subprocess.run(["python3", "temp/_session_scratch_s292/e291_table.py"], capture_output=True, text=True, check=True).stdout.split("\n\n")[0]
now = time.strftime("%H:%M")
txt = f"""
## 5. Result (2026-11-15 {now}; E291 arms c = recipe twin and b = prior trained with corrupted counts, seeds 0 and 1, VAL 873 frames / 22 scenes; arm a (trained clean) was NOT run today; TEST untouched; by hand: replay of the dumps with `jdiag.py boot_pooled` through `run.py` jobs `E291.rule_*`, the rule of section 0 written before the run)
**Token by the rule: G- with R+ = TIE (no measurable gain, harmless) - and the separate kill line (5) is MET: the gain of b over the twin is also produced by a SHUFFLED table. Answer to the question: with a road-level lane-count table built from the GT end clusters (an UPPER BOUND, not a real map) the pair head gains at most +0.0035 topo F1 over its twin, and the content of the table is worth about +0.001 - 0.0014 of that (clean minus U +0.0010; clean minus s about +0.0014, difference of points): the rest is the presence and training of the prior head, or retraining noise.**
**The card's clauses (snapped topo F1, mean over the two seeds, paired scene bootstrap with ONE scene draw per resample for both seeds, 1000 resamples; the interval covers scene sampling, NOT retraining noise):**

| clause | difference | pooled point [95 %] | seed 0 / seed 1 | reading |
|---|---|---|---|---|
| **G** gain with the clean table: b clean - twin | snapped | **+0.0035 [+0.0019, +0.0054]** | +0.0049 / +0.0022 | point below +0.01 and the upper end below +0.01: **G-**; the unsnapped read-out does not lose (+0.0014 [-0.0015, +0.0047], non-inferiority holds) |
| **R** robustness: b at 30 % wrong counts - twin | snapped | **+0.0034 [+0.0020, +0.0051]** | +0.0047 / +0.0021 | lower end above -0.01: **R+** |
| **C** the count itself: b clean - b U (all counts unknown, geometry kept) | snapped | **+0.0010 [+0.0001, +0.0018]** | +0.0007 / +0.0012 | interval excludes 0 (by 0.0001): C+ by the letter, a tenth of a point: the count is worth about 0.001 |
| **kill (5)** SHUFFLED table s (another frame's table) - twin | snapped | **+0.0021 [+0.0005, +0.0040]** | +0.0035 / +0.0007 | lower end above 0: **MET** |
| **kill (5)** sc (own geometry, another frame's counts) - twin | snapped | **+0.0029 [+0.0012, +0.0049]** | +0.0041 / +0.0017 | lower end above 0: **MET** |

The kill line says 'the input would carry something other than the table's count': b gains over the twin when it is given a WRONG table too (s +0.0021, sc +0.0029 against clean +0.0035), so almost all of b's gain is not caused by the table's content. **The twin is a low draw on seed 0** (twin minus the stored E312 snapped -0.0035 on seed 0, +0.0006 on seed 1; the card's 0.005 limit holds): against the STORED E312 the clean gain is +0.0013 / +0.0027 and the s read-out -0.0001 / +0.0013; the retraining noise of the recipe (two seeds of E312 differ by 0.004) is of the size of every effect here, and the scene bootstrap does not see it. **Numbers (topo F1 and the guards, VAL):**

{tab}

**Where the gain sits (analysis.py splits, pooled over seeds, snapped, b clean - twin):** frames whose junction arms all have one lane +0.0047 [+0.0029, +0.0067] (268 frames), frames with a multi-lane arm +0.0033 [+0.0013, +0.0056] (558 frames); arms within 25 m +0.0036, 25 - 40 m +0.0060 [+0.0027, +0.0114] (42 frames; seed 1 -0.0016), beyond 40 m +0.0010. A count prior can only help where the count is above one: the gain is LARGER where the count is one, i.e. it is not a count effect. **Does the gate know when the count is wrong (analysis.py gate, b at 30 % wrong against b clean):** the term-over-token ratio is 0.077 / 0.091 where the count is right and 0.090 / 0.105 where it is wrong (seed 0 / 1), correlation with 'the count is right' -0.139 / -0.109: the gate does NOT distinguish right from wrong counts (slightly larger when wrong), the prior head learned to ignore the content, as a model trained with corrupted counts is free to do.
**Predictions (written before; guesses):** clean +0.008 over the twin: MISSED (+0.0035); at 30 % wrong counts -0.003: MISSED (+0.0034, the gain stays); all counts unknown +0.002: the U read-out is about +0.0025 over the twin (G minus C, difference of points): HIT. Likely token at the stated prediction was INCONCLUSIVE: the data say TIE with the kill line met.
**Cost:** 577 prior parameters (0.02 % of the 2.74 M), negligible GMACs (a gated MLP on the keypoint tokens), ms per frame not measured (the features are built batch-wise in torch from detached keypoint positions: a few launches); the table needs a SD map with the number of lanes per direction at every junction arm, a hard association (the nearest arm within 10 m and 60 degrees, an argmin: a named step) and a presence gate.
**Pros and cons for the discussion (the lead's question: the lane count from the map as a prior, given that maps can have gross errors):** *pros:* cheap (577 parameters), harmless under map errors in this build (30 % wrong counts do not cost anything: the head ignores them), *cons:* with a GT-derived upper-bound table the measurable gain is at most +0.0035 of which about +0.001 is the count; the gate cannot tell right from wrong counts, so the head does not exploit the table (and a head that does exploit it, arm a, would be exactly as bad as the map when the map is wrong: not run); the build needs the hard association step; real maps have errors beyond the 30 % wrong counts modelled here (missing arms, shifted geometry) and the clean row is already an upper bound; so there is no case for taking the count from the map on this evidence. **Open:** arm a (trained clean) would show whether a head that trusts the table gains more on clean counts and what a wrong table then costs (the harm side of the discussion); the position of the arm axis (the mean of the GT end centroids) lets the geometry features act as a position oracle, which C separates only partly (+0.0010); a real SD map (OSM-like) was not used.
**Process:** `c_s0` was restarted from step 0 after a WSL out-of-memory error at step 230 (same seed, same data order); the dumps of one model come from one loader pass (6 - 7 read-outs); all jobs are `run.py` jobs (`E291.c_s* / b_s* / dump_* / card_* / rule_*`); the code is `training/grid/jprior.py` and `jdiag.py boot_pooled` (commit 7950b92ccd) plus staged copies of four digested files (`temp/_session_scratch_e291`, installed after the E343 freeze, or never: nothing to install if the idea is dropped); `rule_gate_s0` exited 2 after printing its result (cause not checked).
"""
s = s.rstrip("\n") + "\n" + txt
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
