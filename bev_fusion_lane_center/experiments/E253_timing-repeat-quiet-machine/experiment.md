# E253 Timing repeat of the chain on a quiet machine (E243 session 2)

The card (read by `experiment.py` and `decide.py`; schema `autoresearch/schemas/experiment.schema.json`):

```json card
{
 "id": "E253",
 "card": "H20.S1",
 "title": "timing repeat on a quiet machine (E243 was invalid): with numba assembly and fork pruning the 192 / 96 fast variant j1 is no faster than the 288 / 144 champion without time recursion; recall variants cost +2 to +3 ms",
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
   "recipe": "wsl bash tools/env/_py14.sh p18_latency.py --name e123_D7_r3_j1 --M 192 --K0 96 --rec 0 --union 2 --fork_prune 0.3 --n 200 --tag e253_j1_r<k>, 6 replicates k = 1..6 interleaved with the other arms",
   "base": "e123_D7_r3_j1",
   "note": "fast variant j1; fork pruning 0.3 as in the stored j1_loop18 timing"
  },
  "c": {
   "recipe": "same command with --name e123_D7_r3_f1 --M 288 --K0 144 (champion without time recursion, pruning 0.3)",
   "base": "e123_D7_r3_f1",
   "note": "stored: 17.54 / 18.23 / 17.76 ms (M288_rec0_u2_fp0.3_f1_loop18 and _rea / _reb)"
  },
  "b": {
   "recipe": "e18_D7_j1eq, M 192, K0 96, rec 0, fork_prune 0.4, n 200, 6 interleaved replicates",
   "note": "plain fast variant; stored 17.71 / 18.73"
  },
  "d": {
   "recipe": "e18_D7_j1rc, M 192, K0 96, rec 0, fork_prune 0.4, n 200, 6 interleaved replicates",
   "note": "fast recall variant; stored 19.6 - 21.8"
  },
  "e": {
   "recipe": "e123_D7_r3_rc1, M 288, K0 144, rec 0, fork_prune 0.4, n 200, 6 interleaved replicates",
   "note": "recall variant 288 / 144; never timed"
  },
  "f": {
   "recipe": "e123_D7_r3_f1, M 288, K0 144, --rec 1, fork_prune 0.3, n 200, 6 interleaved replicates",
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
 "budget_gpu_h": 0.7,
 "kill": "NEW vs E243: a replicate is void when the 1-minute load average is above 3 at its start (CPU guard in temp/loop20/run_e253.sh), besides: discard and repeat a replicate if nvidia-smi shows another GPU process or GPU utilisation above 5 % before the start (the stored j1 pair 19.71 vs 17.26 ms is such a case: a 2.4 ms spread inside one configuration); stop at 0.6 h; a replicate with less than 200 timed frames is void",
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
| id / card / title | E253 / H20.S1 / E243 repeated; the session of loop 19 was invalid (machine 2 - 3 ms above the stored values, spread > 1.5 ms in 5 of 6 configurations) |
| kind | variant (timing only, deterministic checkpoints) |
| metric, direction, band | ms_per_frame, lower, 1.0 ms |
| prediction | j1 (a) minus champion without recursion (c): -0.3 ms (TIE expected) |
| arms | a j1 192/96 fp 0.3, c f1 288/144 fp 0.3, b j1eq, d j1rc, e rc1 288/144, f f1 with time recursion; six interleaved rounds, fresh process per replicate, 200 timed frames |
| rule | KEEP if a is faster than c by >= 1.0 ms with all replicate pairs on the same side; CUT if slower by > 1.0 ms; else TIE; asm_ms guard 1.0 |
| budget, kill | 0.7 GPU h exclusive (about 45 min wall); void a replicate when GPU is busy or the 1-minute load average is above 3 (new guard) |
| evaluate | `temp/loop20/analyse_e253.py` (medians per configuration, spread, reproduction of the stored values) |

## 1. Card
| | |
|---|---|
| lens | system review (measurement protocol) |
| group | A1 speed |
| sources | E243 (invalid session: order only j1 < f1 < j1eq < rc1 < j1rc < f1 + recursion), loop-18 timings, D4 (cost per variant) |
| status | proposed |

- **Statement**: with numba assembly and fork pruning the fast variant j1 (192 / 96) is not faster than the champion without time recursion (288 / 144); the recall variants cost +2 to +3 ms; the time recursion costs about 6 ms.
- **Mechanism**: the builder is a small part of the chain; the assembly (CPU, numba) and the point count dominate, and the 192 / 96 variant yields the same number of vertices per frame.
- **Prediction** {value -0.3 ms, tolerance 0.5 ms, class "timing difference of two chains with the same vertex count"}: a - c = -0.3 ms (decide.py sign: lower is better, so the improvement is +0.3 ms and the clause is read as a - c <= -1.0 ms); j1rc - j1eq = +2.5 ms (tolerance 1.0); f1 time recursion minus f1 = +6 ms (tolerance 1.5); each of a, c, f reproduces its stored value within 1.0 ms. The three bundled claims are scored separately in the summary.
- **Decision rule**: as in section 0; the session is valid only when (1) at least five valid replicates exist in each of the six configurations (voids are repeated in at most two repair rounds), (2) the replicate SD is <= 0.6 ms in at least five of six, (3) a, c and f reproduce their stored values within 1.0 ms (rc1 has no stored value and is judged on SD only); otherwise INVALID and the clause is not read. A TIE again means "j1 is not a speed variant".
- **Test design**: `temp/loop20/run_e253.sh` (E243 driver plus guards: GPU idle, WSL 1-minute load average <= 3, Windows total CPU <= 20 % through powershell.exe; load, Windows CPU, GPU clock and temperature are recorded per replicate in `m_<arm>_r<k>.txt`), run LAST in the loop with nothing else scheduled; other chats were asked to stay off the machine, which is a request and not a lease (PLAN N1); analysis `temp/loop20/analyse_e253.py` (replicate SD, session validity, stored-value reproduction).
- **Cost / kill**: 45 min wall, exclusive; stop at 1 h.
- **Inputs**: ready.
- **Referee** (2026-10-26, ar-referee): REVISE (minor, then ACCEPT; the question is half of the objective and E243's invalid session is exactly why it is repeated; the decision rule can come out against the idea). Revise: (1) GUARD: `/proc/loadavg` inside WSL does not see Windows-side load (other chats' Windows processes, browser, the pending tools chat), is a 1-minute average that lags, and `nvidia-smi --query-compute-apps` often lists nothing under WSL; record per replicate the Windows CPU utilisation (e.g. `powershell Get-Counter '\Processor(_Total)\% Processor Time'` called from WSL) and CPU clock, and void above a stated threshold; also log GPU clock / temperature (N7) so a drift over the rounds is visible. (2) SPREAD CRITERION: 'spread <= 1.5 ms (max - min) in >= 5 of 6' was written for four replicates; with six rounds max - min grows with n: use the SD or IQR of the replicate medians (e.g. SD <= 0.5 ms) and state it before the run; add the reproduction clause to the VALIDITY rule (E243 had both failures: a session with a good spread but all medians +2 ms is still not quotable), and note that rc1 (e) has no stored value, so its reproduction cannot be checked. (3) VOIDED replicates are skipped and never repeated; the 'all pairs on the same side' clause then has fewer pairs: append one repeat round per void (max 2) or require >= 5 valid pairs for a-c. (4) Prediction as {value, tol, class}: a - c = -0.3 ms (tol 0.7) is an improvement-sign question for decide.py (E243 referee point 3: write +0.3 as improvement or label the verdict manual); the three bundled claims (j1 vs f1, recall +2 - 3 ms, recursion +6 ms) each get their own check in the summary. (5) The TIE cannot fail the idea 'j1 is not faster' by more than the band: state that a -0.5 ms result with all pairs negative (as in E243) is again TIE, and what that means for D4 (comparison at equal quality per ms, not 'fast variant'). (6) 'Other chats asked to stay off the machine' is a request, not a lease: the run must wait for the loop to end (it is planned LAST) and `schedule.py` must hold no other item; no WSL job, no browser, no evaluation during the 45 min. (7) Paths: the card says `analyse_e253.py` in temp/loop20/, the driver header says temp/loop19/analyse_e243.py; fix. Cost 45 min and the 1 h stop are realistic (36 replicates x ~1.2 min; voids add time).

## 3. Summary (2026-10-27, run E253.t, 07:20 - 07:38; `temp/loop20/run_e253.sh`, analysis `temp/loop20/analyse_e253.py`)
- **Decision: TIE, session valid.** j1 minus the champion f1 (both pruning 0.3, no recursion) = -0.23 ms (median of the six per-round differences: -0.21 / -0.68 / -0.83 / +0.21 / -0.07 / -0.58); the rule needs -1.0 ms for KEEP. Prediction -0.3 ms (tolerance 1.0): hit. Assembly guard (a - c <= +1.0 ms): passed. j1 is not a speed variant; at equal ms the cheaper arm wins on quality per ms (D4 evidence, no decision).
- **Session validity (card rule, all met):** 6 of 6 valid replicates per configuration (no replicate voided by the guards; one 'machine busy' wait of 20 s), replicate SD 0.44 / 0.29 / 0.21 / 0.39 / 0.26 / 0.76 ms (at most 0.6 in 5 of 6), stored loop-18 values reproduced within 1.0 ms for a, c, f (new 18.52 / 18.75 / 24.85 against 17.15 - 18.10 / 17.54 - 18.23 / 24.20). The machine was quiet: the BEV chat's t1_r4 training had ended 07:19, GPU 0 %, load average 0.9.
- **Medians (ms per frame, 200 frames, six rounds):** j1 18.52, f1 18.75, j1eq 18.49, j1rc 21.14, rc1 20.81, f1 with time recursion 24.85. Recall weight: j1rc - j1eq +2.65 ms (prediction +2.2); rc1 - f1 +2.06 (prediction +2.5); recursion +6.11. Assembly ms: 7.5 - 7.8 for the rec-0 arms without recall, 8.8 - 8.9 for the recall variants. The invalid E243 session had the same order with absolute values 2 - 3 ms higher.
- **Weaknesses:** one session, one machine state; the recursion configuration has the widest spread (SD 0.76 ms, one outlier at 26.7 ms); the fresh-process replicates measure the chain at batch 1 with cached features, not the full system.
- **Cost:** 18 minutes on the idle machine; it had to wait five hours for another chat's training (N1 / D19).
