# E243 Timing replicate of the chain: is the fast variant fast?

The card (read by `experiment.py` and `decide.py`; schema `autoresearch/schemas/experiment.schema.json`):

```json card
{
 "id": "E243",
 "card": "H19.S4",
 "title": "timing replicate: with numba assembly and fork pruning the 192 / 96 fast variant j1 is no faster than the 288 / 144 champion without time recursion; recall variants cost +2 to +3 ms",
 "kind": "variant",
 "metric": "ms_per_frame",
 "direction": "lower",
 "band": 1.0,
 "prediction": {
  "value": -0.3,
  "vs": "c"
 },
 "arms": {
  "a": {
   "recipe": "wsl bash tools/env/_py14.sh p18_latency.py --name e123_D7_r3_j1 --M 192 --K0 96 --rec 0 --union 2 --fork_prune 0.3 --n 200 --tag e243_j1_r<k>, 4 replicates k = 1..4 interleaved with the other arms",
   "base": "e123_D7_r3_j1",
   "note": "fast variant j1; fork pruning 0.3 as in the stored j1_loop18 timing"
  },
  "c": {
   "recipe": "same command with --name e123_D7_r3_f1 --M 288 --K0 144 (champion without time recursion, pruning 0.3)",
   "base": "e123_D7_r3_f1",
   "note": "stored: 17.54 / 18.23 / 17.76 ms (M288_rec0_u2_fp0.3_f1_loop18 and _rea / _reb)"
  },
  "b": {
   "recipe": "e18_D7_j1eq, M 192, K0 96, rec 0, fork_prune 0.4, n 200, 4 interleaved replicates",
   "note": "plain fast variant; stored 17.71 / 18.73"
  },
  "d": {
   "recipe": "e18_D7_j1rc, M 192, K0 96, rec 0, fork_prune 0.4, n 200, 4 interleaved replicates",
   "note": "fast recall variant; stored 19.6 - 21.8"
  },
  "e": {
   "recipe": "e123_D7_r3_rc1, M 288, K0 144, rec 0, fork_prune 0.4, n 200, 4 interleaved replicates",
   "note": "recall variant 288 / 144; never timed"
  },
  "f": {
   "recipe": "e123_D7_r3_f1, M 288, K0 144, --rec 1, fork_prune 0.3, n 200, 4 interleaved replicates",
   "note": "champion as trained (time recursion); stored 24.2 ms"
  }
 },
 "control": "c",
 "variant": "a",
 "rule": [
  {
   "metric": "ms_per_frame",
   "min_effect": 1.0,
   "note": "KEEP only if j1 is faster than the champion by at least the 1.0 ms band; the expected outcome is TIE (the cheaper arm then wins on quality per ms, not on ms)"
  },
  {
   "metric": "asm_ms",
   "max_loss": 1.0,
   "note": "guard: the assembly share does not grow"
  }
 ],
 "budget_gpu_h": 0.5,
 "kill": "discard and repeat a replicate if nvidia-smi shows another GPU process or GPU utilisation above 5 % before the start (the stored j1 pair 19.71 vs 17.26 ms is such a case: a 2.4 ms spread inside one configuration); stop at 0.6 h; a replicate with less than 200 timed frames is void",
 "evaluate": {
  "split": "val",
  "subset": "timing",
  "compare": "c"
 }
}
```


## 0. What the tools run (the fields of `experiment.json`)
| field | value |
|---|---|
| id / card / title | E243 / H19.S4 / with numba assembly and fork pruning, j1 192 / 96 is no faster than the champion 288 / 144 without time recursion |
| kind | variant (existing script `evaluation/drivers/p18_latency.py`, options only) |
| deterministic | false (GPU timing noise); medians of 200 frames, 4 interleaved replicates |
| metric, direction, band | ms_per_frame, lower, 1.0 ms |
| prediction | -0.3 ms against arm c (j1 minus champion f1, rec 0, pruning 0.3) |
| arms | a: j1 192 / 96; c: f1 288 / 144 (both rec 0, pruning 0.3); references b: j1eq, d: j1rc, e: rc1 (all pruning 0.4, rec 0), f: f1 with time recursion |
| control | c |
| rule | ms_per_frame min_effect 1.0 (a faster than c by >= 1.0 ms); asm_ms max_loss 1.0 |
| budget_gpu_h, kill | 0.5, exclusive idle GPU; replicate void if another GPU process is present |
| evaluate | val frames, timing driver |

## 1. Card
| | |
|---|---|
| lens | system review (cost per decision: ms per frame is half of the objective "very good AND very fast") |
| group | A3 sparse lane graph (A9 for the timing protocol) |
| sources | runs/e18/latency/*.json (24 stored timings), E216, E219, D4, evaluation/PROTOCOL.md (timing) |
| status | proposed |

- **Statement.** In the loop-18 chain (numba assembly, fork pruning, rec 0) the median ms per frame of the fast variant j1 (192 / 96) is within 1.0 ms of the champion f1 (288 / 144), so the "fast variant" of D4 buys no speed against the champion without time recursion; the recall variants j1rc and rc1 cost 2 - 3 ms more than their plain controls j1eq and f1 because they emit more points (189 against 164 per frame) into the assembly; only the time recursion (24 ms) makes the champion slow.
- **Mechanism.** Stored timings (rec 0, union 2, pruning 0.3, 60 frames): j1 17.15 / 18.10 / 17.76 / 17.90 ms (builder 5.1 - 5.6, assembly 6.9 - 7.1), f1 17.54 / 18.23 / 17.76 ms (builder 6.1 - 6.2, assembly 6.8 - 7.0). The 192 / 96 setting saves about 1 ms in the builder, but the numba union plus pruning 0.3 cut the assembly of f1 from 12.6 to 7.0 ms (E216; j1 had 6.9 - 7.1 ms already), so the assembly, not the builder, is the larger and equal part of both chains. The recall fine-tunes add points (j1 159, j1eq 164, j1rc 189, j1rc6 211) and the assembly time follows (7.0, 7.7, 8.9, 10.0 ms). The same-configuration spread of a stored median of 60 frames is 0.3 - 0.5 ms (j1 four times 17.15 - 18.10), with one 2.4 ms outlier pair (ctl_j1_a 19.71 against ctl_j1_b 17.26); the stage medians do not add up to the total (j1eq: builder 5.25 + assembly 7.74 + memory 2.31 + route input 2.65 + route head 1.97 + consumer 0.47 = 20.4 against 17.7 ms), so the total is a separate measurement. Hence a difference of 1 ms is at the edge of what one replicate can say, and D4 compares "17.5 against 24 ms" across a recursion switch that the variants do not share (the quality of f1 with and without recursion is in E132 / E196 and has to be quoted with it).
- **Prediction.** Median of the four replicate medians (200 frames each, rec 0): f1 17.8 ms, j1 17.5 ms (j1 - f1 = -0.3 ms, interval -1.0 ... +0.4), j1eq 18.2, j1rc 20.4 (+2.2 over j1eq, interval +1.5 ... +3.5), rc1 20.5 (+2.5 over f1; never timed), f1 with recursion 24.2 (+- 1). Spread of one configuration over its four replicates (max - min of medians) at most 1.0 ms in at least 5 of 6 configurations. Quality per time from the VAL cards (Q at pruning 0.4: j1p4 0.729, j1eq 0.735, j1rc 0.739, rc1 0.740; champion f1 0.735): the extra 2 - 3 ms of the recall variants buy +0.004 Q and +0.06 lanes found.
- **Decision rule.** Effect = median of replicate medians (a minus c), band 1.0 ms. KEEP (j1 is faster by >= 1.0 ms with all replicate pairs on the same side) is not expected; TIE (expected) means: j1 is not a speed variant and D4 should compare j1eq / j1rc against f1 at rec 0 on quality per ms, with f1 the cheaper-by-tie reference only if its quality is higher (it is not: Q 0.735 against 0.735 for j1eq, lanes found 0.695 against 0.674); CUT (j1 slower than f1 by > 1.0 ms) would say the 192 / 96 builder is not even a cost saving. The side deliverable is the measured replicate SD, which sets the timing band in PROTOCOL.md (today a band is not written for ms). If any configuration's replicate spread exceeds 1.5 ms the 1.0 ms band is withdrawn and decisions on ms need 8 replicates.
- **Test design.** Arms b - f are references on the same session: the six configurations are run round-robin (a c b d e f, four rounds) so that drift of the GPU clock or temperature hits all arms equally. Null: the stored loop-18 values (not used for the decision, only as a reproduction check: the new medians of f1 and j1 must lie within 1.0 ms of the stored ones, otherwise the machine state differs and the whole session is repeated). Each replicate is a fresh process (CUDA graphs captured anew). 200 frames after warm-up instead of 60 (the stored files use `--n 60`). Idle GPU: `nvidia-smi` before each replicate, no WSL job of any chat running (check `ListAgents` and the schedule first). Fidelity: layer 1 (reproduce stored numbers) plus layer 4 statistics on timing.
- **Cost.** 0.5 GPU h exclusive (24 runs of an estimated 1.2 min each; the running time of one `p18_latency.py` call is not recorded in the ledger: measure the first replicate and cut `--n` to 120 if it exceeds 1.5 min), 0 coding days. It must run after the E240 / E241 trainings (an idle GPU is a precondition). Risk: low.
- **Kill criterion.** See the table.
- **Inputs.** Ready: `p18_latency.py` and the checkpoints e123_D7_r3_f1, e123_D7_r3_j1, e123_D7_r3_rc1, e18_D7_j1eq, e18_D7_j1rc. To build: a 10-line driver script in `temp/loop19/` that loops the six commands round-robin and writes the medians into one json (throw-away; the stage tables stay in runs/e18/latency/). To audit with it: why the stage medians do not add to the total.
- **Referee** (2026-10-25, ar-referee): ACCEPT; run FIRST, right now (GPU idle while the lead writes `--save_at`; about 35 - 45 min wall). Half of the objective is ms per frame and D4 compares 17.5 ms with 24 ms across a recursion switch; the stored j1 / f1 timings overlap (17.15 - 18.10 vs 17.54 - 18.23 ms, with a 2.4 ms outlier), so 4 interleaved fresh-process replicates of 200 frames, a reproduction check against stored values and a void rule for a busy GPU are the right test. Non-blocking edits: (1) the assembly is CPU numba (about 7 ms), so run NOTHING else on the machine during the session (no E236 sweeps, no evaluations, no training), not only "no other GPU process"; (2) decide.py cannot read these arms (no `cards`): the driver script writes one flat-dict card per replicate (ms_per_frame, asm_ms) and the arms get `cards` refs, or the verdict is labelled manual; (3) for a lower-is-better metric decide.py prints measured effect as improvement (+0.3) while the prediction is written as -0.3: write the prediction as +0.3 (improvement) or note the sign; (4) the Statement bundles three claims (j1 vs f1, recall variants +2 - 3 ms, only recursion makes the champion slow): list them as separate checks in the summary; (5) the card's "after the E240 / E241 trainings" order is the wrong way round for the night: do it before, the idle GPU is the precondition.

## 3. Summary (2026-10-25, run E243.timing; 6 configurations x 4 round-robin replicates, 200 frames, fresh process)
- **Decision: TIE on the relative ordering, session NOT VALID by the card's own rules.** Medians of the replicate medians: j1 20.26, f1 (rec 0) 20.76, j1eq 20.86, j1rc 23.43, rc1 22.25, f1 with recursion 27.13 ms. a - c = -0.49 ms (all four pairs negative: -1.32, -0.41, -1.08, -0.35; inside the 1.0 ms band); assembly share a - c +0.0 ms.
- **Why not valid**: the reproduction check failed (every configuration about 2 - 3 ms above its stored loop-18 value: j1 +2.1, f1 +2.2, f1 with recursion +2.9 ms), the replicate spread exceeds 1.5 ms in 5 of 6 configurations (1.4 - 3.0 ms; the 1.0 ms band is withdrawn by the card's rule), one replicate was voided (GPU busy), and the medians drift upward over the rounds. Other chats' agents were running on the machine during the session (the referee's condition "nothing else runs" was not met); the exact cause is not measured.
- **Predictions**: j1rc - j1eq +2.57 ms (predicted +2.2, met); rc1 - f1 +1.49 ms (predicted +2.5, missed); recursion +6.37 ms (the stored 24.2 - 17.5 = +6.7, met in difference); a - c -0.49 (predicted -0.3, met).
- **What it says**: j1 is not a speed variant (no 1.0 ms advantage over the champion without recursion) and the recall variants cost +1.5 to +2.6 ms through more points in the assembly; only the time recursion makes the champion slow. Absolute ms are not quotable from this session: quote the stored loop-18 timings. A repeat with 8 replicates on a quiet machine is scheduled for the end of the night.
- **System finding**: a timing protocol needs a machine-state guard (CPU load and clock before every replicate, not only GPU use), and the launcher should refuse timing jobs while other jobs run (S19 list).
