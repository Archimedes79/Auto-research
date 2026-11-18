# E2E cost table - what the pieces of the token champion cost at inference (measured 2026-11-18 05:18 - 05:30)

Measured numbers only; "not measured" where it is. RTX 3060, batch 1, bf16 autocast, CUDA events with a sync in front and behind, no_grad, cached VAL frames, 24 frames x 5 repeats (S2: 24 x 4, S3: 20 x 5) after one untimed pass, the median of the timed frames; the host must read QUIET (load below 3.0 before and after, GPU idle at most 5 % before and after, `jtiming.quiet_verdict`). Every row below carries that verdict. Written by the lane chat for the lead's request (S2 / S3 / ensemble cost) and the report.

## The numbers
| piece (what is timed) | median ms | p90 ms | parameters | against |
|---|---|---|---|---|
| **champion path**: B' (QGraph `e292_a_s0`) + ONE token head (`e373_cb_s0`), whole path | **16.21** | 19.89 | 2,323,143 (B' 2,196,012 + head 127,131) | - |
| B' alone | 11.20 | 15.53 | 2,196,012 | - |
| one token head alone (probabilities) | 5.94 | 7.91 | 127,131 | - |
| **e2**: two heads in sequence (`e373_cb_s0` + `e367_b25_s0`), whole path | **21.84** | 27.77 | 2,450,274 | champion +5.63 ms (35 %) |
| e2: two heads STACKED (`torch.func.stack_module_state` + `vmap`), whole path | 25.19 | 31.54 | 2,450,274 | champion +8.98 ms; slower than the sequence (the head is latency-bound at batch 1); output equals the two single forwards to 1.0e-06 |
| e2 heads only: sequence / stacked (probabilities) | 12.13 / 14.92 | 17.52 / 16.31 | 254,262 | one head 5.94 |
| **S2** (E368): the head WITH the past block `r` (`e368_r_s0`), head alone, stream order | **7.88** | 9.75 | 144,547 (block 17,416) | block cost **+1.23 ms** over `z` (6.66 ms, 127,131); the first frame of a scene (no past) 7.66 ms |
| **S3** (E374): the head WITH the SD-map block `m` (`e374_m_s0`), head alone | **7.95** | 9.19 | 147,395 (block 20,264) | block cost **+1.70 ms** over `z` (6.26 ms, 127,131); about 154 map samples per frame |
| **S4** (E372): B' node head trained under the head's loss | not measured here | - | - | expected to cost what the champion path costs (the same modules, B' + head, trained jointly: other weights, not other layers); NOT timed, its timing is not part of tonight's chain |

## Reading
- **e2 does not meet the cost bar of the E379 gate (D106): the stacked whole path is 25.19 ms and even the sequence is 21.84 ms, both above the 19.7 ms the card names (E324 s, quoted from the card, not re-measured tonight).** The ensemble buys topo F1 +0.0113 / +0.0065 (card section 3) at +5.63 ms and +127,131 parameters; the lead decides (D106).
- S2 and S3 cost +1.23 / +1.70 ms on the HEAD (the S2 card predicted +1 - 2 ms: matches) and neither block was adopted (E368 and E374 TIE by their written rules).
- The champion path is 16.21 ms here (69 % of it is B' alone, timed separately); the head is the cheaper part.

## How the numbers were made, and the first (not-QUIET) runs
Run ids `E379.time_e2b` (`token_time_ens.py`, the card's registered rule), `diag.s2_timing_b` (`temp/e368/time_s2.py`, scratch), `diag.s3_time_m_b` / `diag.s3_time_z_b` (`jmap_eval.py time` through `temp/e374/time_s3_aged.py`, scratch); files `e373_cb_s0/token_timing_ens_20261118T042140Z.json`, `e368_r_s0/past_timing_20261118T042441Z.json`, `e374_m_s0/map_timing_20261118T042740Z.json`, `e374_z_s0/map_timing_20261118T043035Z.json`.
Hosts: e2 host load (1 min) before 1.06 after 0.9, limit 3.0; GPU before util max 1 %, after util max 0 % : QUIET; S2 host load (1 min) before 1.17 after 0.99, limit 3.0; GPU before util max 0 %, after util max 0 % : QUIET; S3 m host load (1 min) before 1.16 after 1.07, limit 3.0; GPU before util max 1 %, after util max 0 % : QUIET; S3 z host load (1 min) before 1.06 after 0.97, limit 3.0; GPU before util max 1 %, after util max 0 % : QUIET.
**The first runs of the same night (05:09 - 05:14: `E379.time_e2`, `diag.s2_timing`, `diag.s3_time_m`) were flagged NOT QUIET only because the tools took the GPU after-sample at once: nvidia-smi averages its utilisation over a window, so the sample read the process's own last kernels (65 - 70 %) on an idle box; `jtiming.py`'s own protocol waits 2.5 s first (`time.sleep(2.5)`), the timing commands of the other tools did not.** The repeat with the wait (`token_time_ens.after_state`, the two scratch scripts) reads QUIET, and the milliseconds agree with the first runs: e2 whole path 16.29 / 22.02 / 25.30 ms (first run: one head / sequence / stacked) against 16.21 / 21.84 / 25.19 (repeat), S2 block cost +1.29 against +1.23 ms, S3 m 8.60 against 7.95 ms. `jtoken_eval.py timing` has the same missing wait until the unit of the token-head owner lands: a NOT QUIET flag right after its timed passes may be this artefact, the milliseconds are not affected.
Limits: one cache seed (0), one pair of heads for e2, 24 / 20 frames: the p90 is wider than the median (a stall in a few frames); two quiet repeats differ by about 0.1 - 0.4 ms (the two e2 runs by 0.18 ms on the sequence). S4's own cost and the timing of the K 384 variants (E383) are not in this table.
