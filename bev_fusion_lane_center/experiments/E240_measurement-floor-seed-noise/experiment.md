# E240 Measurement floor: seed noise of the recall fine-tune

The card (read by `experiment.py` and `decide.py`; schema `autoresearch/schemas/experiment.schema.json`):

```json card
{
 "id": "E240",
 "card": "H19.S1",
 "title": "measurement floor: seed noise of the recall fine-tune on Q, topo F1, lanes found, R@1 far (j1rc against j1eq, four seeds per arm)",
 "kind": "variant",
 "metric": "Q",
 "direction": "higher",
 "band": 0.01,
 "prediction": {
  "value": 0.006,
  "vs": "c"
 },
 "arms": {
  "a": {
   "recipe": "j1rc = p16_train.py --design D7 --init e123_D7_r3_j1 --u 4 --lr 0.0005 --stem_lr 0.3 --w rec=3.0 (the exact E219.j1rc command), seeds 0 / 1 stored (e18_D7_j1rc, e18_D7_j1rcs1), seeds 2 / 3 new (e24_D7_j1rcs2, e24_D7_j1rcs3), fork_prune 0.4",
   "base": "e123_D7_r3_j1",
   "seeds": [
    0,
    1,
    2,
    3
   ],
   "note": "all four evaluated with evaluate.py (new registry names e24_j1rc_s0..s3) so that topo_F1 and R@1_far are on the cards; stored seeds 0 / 1 are re-evaluated, not re-trained | pair 3 trained in loop 20 on the stored distance map (LCR_DIST_DIR), read with four pairs",
   "names": [
    "e24_j1rc_s0",
    "e24_j1rc_s1",
    "e24_j1rc_s2",
    "e24_j1rc_s3"
   ]
  },
  "c": {
   "recipe": "j1eq = the same command with --w rec=1.0 (equal-budget control, E219.j1eq), seeds 0 / 1 stored (e18_D7_j1eq, e18_D7_j1eqs1), seeds 2 / 3 new (e24_D7_j1eqs2, e24_D7_j1eqs3), fork_prune 0.4",
   "base": "e123_D7_r3_j1",
   "seeds": [
    0,
    1,
    2,
    3
   ],
   "note": "registry names e24_j1eq_s0..s3 | pair 3 trained in loop 20 on the stored distance map (LCR_DIST_DIR), read with four pairs",
   "names": [
    "e24_j1eq_s0",
    "e24_j1eq_s1",
    "e24_j1eq_s2",
    "e24_j1eq_s3"
   ]
  }
 },
 "control": "c",
 "variant": "a",
 "rule": [
  {
   "metric": "Q",
   "min_effect": 0.005,
   "note": "same clause as E219: the weight must pay in Q, judged now on four paired seeds"
  },
  {
   "metric": "lanes_found",
   "min_effect": 0.04
  },
  {
   "metric": "topo_F1",
   "max_loss": 0.01,
   "note": "guard"
  },
  {
   "metric": "F1@1.0",
   "max_loss": 0.005,
   "note": "guard"
  }
 ],
 "seeds": [
  0,
  1,
  2,
  3
 ],
 "budget_gpu_h": 1.2,
 "kill": "stop the pair of new fine-tunes if a final training loss deviates by more than 10 % from the stored seeds of the same arm (derail, BN running_var) or GPU memory exceeds 11 GB; a derailed seed is re-run once with the same seed and reported, never dropped silently",
 "evaluate": {
  "split": "val",
  "compare": "c"
 }
}
```


## 0. What the tools run (the fields of `experiment.json`)
| field | value |
|---|---|
| id / card / title | E240 / H19.S1 / seed noise of the fine-tune on Q, topo F1, lanes found, R@1 far (j1rc against j1eq, four seeds per arm) |
| kind | variant (configuration only: the E219 command with two more seeds per arm) |
| deterministic | false (training seeds) |
| metric, direction, band | Q, higher, 0.010 (the band in force; the experiment measures whether it is right) |
| prediction | +0.006 against arm c (j1rc - j1eq on Q, four seeds) |
| arms | a: j1rc (recall weight 3), seeds 0 1 stored + 2 3 new; c: j1eq (weight 1, equal budget), seeds 0 1 stored + 2 3 new |
| control | c |
| rule | Q min_effect 0.005; lanes_found min_effect 0.04; topo_F1 max_loss 0.01; F1@1.0 max_loss 0.005 |
| budget_gpu_h, kill | 1.2 (4 fine-tunes of 13 min alone, two at a time, + 8 evaluations of about 3 min); derailed seed -> re-run once, reported |
| evaluate | val (873), compare c |

## 1. Card
| | |
|---|---|
| lens | system review (measurement floor, ROADMAP P9 / D4 "3 seeds + scene interval") |
| group | A9 evaluation, data, metrics |
| sources | E219 (j1rc, j1rcs1, j1eq, j1eqs1, j1eq8), E215, runs/e18/rc_eval/*.json, runs/eval/champion/val_val250_fresh_*.json, D4 in DECISIONS.md |
| status | proposed |

- **Statement.** The seed-to-seed standard deviation of the 4 u recall fine-tune of j1 (VAL 873, fork pruning 0.4) is at most 0.004 on Q and at most 0.006 on lanes found, so the minimum detectable Q effect (2.8 x SD x sqrt(2 / n) per arm) is at most 0.0085 with the usual two seeds and at most 0.0060 with four, i.e. the written band of 0.010 is a fair price for two seeds and too wide for four; and j1rc still beats j1eq on lanes found by at least +0.04 with all four paired seeds on the same side.
- **Mechanism.** Evaluation is bit-deterministic (three fresh VAL250 evaluations of the champion give identical Q, topo F1, lanes found, R@1 far: runs/eval/champion/val_val250_fresh_fresh1 / fresh2 / golden), so all run-to-run variance comes from the fine-tune seed (head init, batch order, AMP). Six stored pairs give a small spread: Q |difference| of seed pairs 0.0028 (rc1), 0.0035 (j1rc), 0.0045 (j1eq); lanes found 0.002, 0.007, 0.003. These are three pairs only (df = 3), so the SD is not yet known to better than a factor of two. The minimum detectable effect of a paired two-seed comparison is about 2.8 x SD, i.e. 0.008 for Q at SD 0.003, which is why E219 (+0.0077, both seeds positive) ended TIE although the lanes-found effect was 8 times the noise.
- **Prediction.** On VAL 873, pooled over both arms (n = 4 each, df = 6): SD(Q) = 0.0030 (plausible 0.002 - 0.0045), SD(lanes found) = 0.0035 (0.002 - 0.006), SD(topo F1) = 0.003 (0.002 - 0.005), SD(R@1 far) = 0.006 (0.003 - 0.010; far recall is a tail metric and should be the noisiest). Arm difference j1rc - j1eq: Q +0.006 (0.002 - 0.010), lanes found +0.062 +- 0.012, topo F1 within +-0.005. The first value is the 2-seed E219 result +0.0077 shrunk towards zero (E219.j1eq8 showed the controls drift by 0.004).
- **Decision rule.** The four clauses of the rule above through `decide.py --experiment E240` (paired seeds i against i, scene bootstrap from `evaluate.py --compare`). Verdict meaning for the weight: KEEP = j1rc is better than j1eq on Q by more than the band with four agreeing seeds; TIE = inside 0.010 (the expected outcome, Q +0.006 < 0.010: then the cheaper arm j1eq wins on Q and j1rc is justified only by lanes found, which the second clause tests). The side result decides the improver: if pooled SD(Q) <= 0.0035 and SD(lanes found) <= 0.005, propose the metric-specific bands Q 0.006 and lanes found 0.015 (written as a DECISIONS entry; PROTOCOL.md and project.json are protected) and the 2-seed rule stays; if SD(Q) >= 0.005 the 0.010 band stays and screening needs 3 seeds. In both cases `ar metrics` reports the measured SD next to every band.
- **Test design.** Arms and controls as in the table. Seed twin = the stored seeds 0 / 1 re-evaluated through `evaluate.py` (same checkpoints, so they must reproduce the stored Q to 4 digits: layer 1 check, otherwise the registry entries are wrong). Equal budget = both arms 4 u. Null = j1 itself (e123_D7_r3_j1, pruning 0.4 card j1p4: Q 0.729), so that "extra fine-tuning" and "weight" are separated as in E219. Two fine-tunes run at once (GPU < 11 GB, never three). Seeds 2 / 3 are new training seeds with the same data order seed scheme as seeds 0 / 1. Fidelity rung: layer 4 (VAL 873) because the question is about the rung itself.
- **Cost.** 1.2 GPU h (GPU), 0 coding days, 0.1 day writing four registry entries `e24_j1rc_s0..3`, `e24_j1eq_s0..3` (new names; the protected champion / control entries are not touched). Risk: low. These four fine-tunes are also the base of E241, so launch them with the `--save_at` flag of E241 (code first) or E241 loses its curves.
- **Kill criterion.** See the table: a derailed seed (final loss more than 10 % off) is re-run once and reported; stop everything if the stored seeds 0 / 1 do not reproduce their Q to 4 digits through `evaluate.py`.
- **Inputs.** Ready: the E219 command lines (LEDGER.csv rows E219.j1rc / j1eq), `evaluate.py`, `decide.py --experiment`. To build: the eight registry entries; `--save_at` (E241). Stored seeds 0 / 1 checkpoints exist in runs/e123/e18_D7_j1rc, j1rcs1, j1eq, j1eqs1.
- **Referee** (2026-10-25, ar-referee): ACCEPT (first GPU job of the night, after `--save_at` is in). Directly serves D4 ("3 seeds + scene interval") and fixes the noise band every other card relies on; evaluation is deterministic, so the variance is the training seed; controls, equal budget, null (j1), layer-1 reproduction of stored seeds and a re-run rule for derailed seeds are all there. Non-blocking edits: (1) arithmetic in the Statement: with SD(Q) = 0.004 the minimum detectable effect 2.8 x SD x sqrt(2 / n) is 0.0112 (n = 2) and 0.0079 (n = 4), not 0.0085 / 0.0060; those figures belong to SD 0.003. State the threshold as SD <= 0.003, or correct the MDE; (2) with df = 6 the SD is known only to about a factor 0.65 - 1.9 (90 %): report the point estimate with its upper bound and base the band proposal on the bound; (3) the verdict (j1rc vs j1eq on Q) is expected TIE by design (+0.006 < the band 0.010): the real outputs are the SDs, so put them as numbers into the rule note or a fifth clause for the summary; (4) launch seeds 2 / 3 on the same commit as `--save_at` plus nothing else (no other edit of p16_train.py / p16_loss.py until the four runs have started: a changed loss file between seeds would confound the SD); (5) cost: solo 13 min per fine-tune (786 s in the cards), two concurrent take about 25 min each, so run them sequentially or one training plus one evaluation: wall clock about 55 min training + evaluations, 1.2 GPU h is fine.

## 3. Summary (2026-10-26 / 27, runs E240.a2 / c2 and E240.a3 / c3 (seeds 2 and 3 new; seeds 0 / 1 are the stored E219 runs), E240.ev1; VAL 873, fork pruning 0.4, tau 0.7, `evaluate.py`; script `temp/loop19/e240_sd.py`)
- **Decision: TIE by the tool, four pairs** (`decide.py --experiment E240`: Q a - c = +0.0062, inside the band 0.010; lanes found +0.0596 ok, topo_F1 -0.0012 ok, F1@1.0 +0.0041 ok). Loop 19 closed with three pairs (pair 3 cancelled on the shared machine); pair 3 was trained in loop 20 (E240.a3 / c3) and the card is read with four pairs as designed. Deviation, labelled: pair 3 was trained on the stored (TF32-damaged) distance map, pinned by `LCR_DIST_DIR`, because D14 switched the loader default to the exact map after pairs 0 - 2; the four pairs share one map. The pair-3 numbers sit inside the range of pairs 0 - 2 (Q j1rc 0.7391, j1eq 0.7316).
- **Layer 1**: the stored seeds 0 / 1 reproduce E219 through `evaluate.py` (E240.ev1 PASS).
- **Measurement floor (the real output)**, seed SD (n = 4 per arm): Q j1rc 0.0030, j1eq 0.0020 (pooled about 0.0025); lanes found 0.0034 / 0.0042; R@1 far 0.0043 / 0.0022; topo_F1 0.0013 / 0.0008; F1@1.0 0.0018 / 0.0004; vert100 0.09 / 0.06. SD of the paired difference: Q 0.0043, lanes found 0.0059, R@1 far 0.0062. The band 0.010 for Q is about 4 pooled SDs and 2.3 paired SDs; the SD of an SD with n = 4 is still about 40 %, so these are estimates (the three-pair SDs were 0.0036 / 0.0022: a fourth pair moved them by 0.0006 at most).
- **j1rc - j1eq per pair**: Q +0.0038 / +0.0117 / +0.0020 / +0.0075 (mean +0.0062, min +0.0020, max +0.0117), lanes found +0.0621 / +0.0662 / +0.0525 / +0.0578 (mean +0.0596), R@1 far +0.0392 / +0.0523 / +0.0463 / +0.0523 (mean +0.0475); topo_F1 -0.0013 / +0.0004 / -0.0037 / -0.0004 (mean -0.0012). Lanes found and R@1 far have the same sign in all four pairs and are 10.1 (lanes found) and 7.6 (R@1 far) paired SDs (a rough ratio, not a significance statement); the Q edge is positive in all four pairs (+0.0062, standard error 0.0022, t about 2.9 on 3 df) but small and, in three of four pairs, below the band: below the 0.010 decision band, not shown to be zero.
- **Vertex budget**: j1rc has +0.16 vertices per 100 m (15.03 against 14.86); at the matched-vertex slope of E236 (about 0.0025 Q per 0.25 vertices) this is about 0.002 of the +0.0062; that is an estimate from the E236 slope, not a measurement: per pair the vertex gaps are +0.04 / +0.22 / +0.34 / +0.06 for Q gaps +0.0038 / +0.0117 / +0.0020 / +0.0075, so the pairs do not show the slope. E255 later measured the true matched-vertex cost for a different arm (the Poisson loss) and found that the slope is much steeper for a vertex-hungry arm than for the j1 family.
- **Prediction** {value 0.006, tol 0.005, class "Q effect of a fine-tune, median 0.004"}: measured +0.0062: hit (error +0.0002).
- **What it says**: the recall weight 3 is a lanes-found / R@1-far lever (+0.06 / +0.048, every pair), not a Q lever (Q +0.002 to +0.012, mean +0.006, TIE by the tool). This supports the D4 option list: j1rc buys lanes found at +2-3 ms (E243, invalid session, ordering only) and no Q; the decision stays Mirko's (D4).
- **Weaknesses**: n = 4 pairs, three commits (stored code for seeds 0 / 1, loop-19 code for seed 2, loop-20 code for seed 3; default-off differences only), pair 3 on the stored map while the loader default is now the exact map, SDs are estimates.
- **Next / system**: metric-specific bands (Q SD about 0.0025, lanes found about 0.004, R@1 far about 0.003, topo F1 about 0.001) are now measured on four pairs; the proposal to replace the single band 0.010 by per-metric bands goes into the next DECISIONS round together with D4 (the rule file `project.json` / `decide.py` bands are protected).
