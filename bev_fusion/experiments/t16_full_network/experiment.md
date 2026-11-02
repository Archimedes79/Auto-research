# t16 full network - does the complete chain (every built block, five-layer framework, GT v8) train, and what is its baseline? (bring-up card, 2026-11-02, before any run)

## 0 Statement, referee, status
- **Order (D44, Mirko 2026-11-02, word for word in `experiment_execution/DECISIONS.md`):** generate a full network with fusion; train it first on a small set; time the training code; probe the outputs of all modules ("does it make sense?"); find errors and fix them; train again until fixed, with timing; then train on the large set and establish a performance baseline; make a report with result videos and images; afterwards module improvements and the repeat of old experiments on the new system and GT v8.
- **What this card is.** A bring-up with gates, not a hypothesis test. The small runs are diagnostics (`--no-card`, they feed no verdict). The one run that counts is the large run: it becomes the baseline row of the new system. Everything it is compared with is labelled by hand ("by hand: single seed, bring-up baseline, no verdict"), because one seed never gives KEEP or CUT (project rule).
- **Status.** Draft; config `training/configs/t16_full.json` and preset `training/configs/presets/small_set.json` written; static check passes; CPU gate `tests/test_slots.py` (with the full combination) passes. No GPU run yet: the run snapshot carries the working tree, so nothing starts before the other chats have committed (D44 (2)).
- **TEST is untouched** (VAL only; hard rule 1).

## 1 The network (one config, every block that is built and has a passing gate)
| slot | kind in `t16_full.json` | source / gate |
|---|---|---|
| camera encoder | `camera_features` "dual": cached DualBEV heads (stride 16) + BEVFusion map-trained features (stride 8), two lifts PD merged by a 1x1 conv, 3 frames | t10 chain; `test_dual_camera_geometry.py` |
| camera temporal | `window_memory` (window of the last frames + world-aligned static memory) | t2 / t10 |
| radar encoder | `delta_slots` R, `c_rad` 64, side loss `radar_aux` 1.0 -> 0.05 | Radar Encoder chat, `test_radar_encoder.py`, `test_radar_slot.py` |
| radar temporal | `window_memory` with the Doppler static gate (`rad_static_gate`) | built, **never run**: the least tested block of the chain; its window part and velocity head get no gradient without past radar frames (named in `test_slots.py` KNOWN_UNUSED) |
| fusion | `concat` + modality dropout `p_drop` 0.2 | t2 (all fusion kinds TIE) |
| backbone + neck | DualBEV trunk width 1.0, detail path 32, pre-trained, `head_channels` 96, `c_fuse` 80 | t1 r4 / t2 |
| heads | maps (5 areas, 2 lines, 2 distance fields, 3 dense objects), radar aux head, object head with `object_direction` and the velocity hurdle `object_moving` | t1 r1a, t11 |
| motion block M | `motion_block` (pyramid 4 / 2 / 1), `box_vel_from_field` | t12 (frozen base KEEP), joint form `t15`; **joint training never run** |
| soft extraction | `dense_outputs` off in training and timing, on for the probes and the report | `extract.py` |

Left out on purpose: `lane_branch` (t7, no gain), `refine` (D43: a separate stage network), `centrelines` (no head design yet), `cam_adapter`, `cam_collapse` zsum, `trunk_kind` second, `fusion` gate / cmx (all TIE or cut). The combination "M + hurdle + R + radar memory" is new: `test_slots.py` builds it, passes both interfaces and sends a gradient to every trainable parameter except the named unused parts (56 s CPU, 2026-11-02).

Recipe: that of `t10_dual_long_s0` (6000 optimiser steps, batch 3 x accumulation 2, lr 3e-4, warm-up 300, Kendall balancing, EMA 0.999, flip 0.5), no `init_from`, `probe` on (grad.<module>, upd.<module>, t_data / t_step in every logged row). Large set = the 428 / 82 scenes of the t10 chain (the BEVFusion features exist for these only); small set = autogt's small set (10 scenes of nuScenes mini: 8 train, 2 val; 323 + 81 keyframes), features linked in `data/cache/small/`.

## 2 Stages and gates
| stage | what | gate (all numbers from `log.jsonl`, `metrics.json`, `status`; written before the run) |
|---|---|---|
| S0 CPU | `tests/test_slots.py`, `startable.py`, `tests/equivalence.py` on the final HEAD (6 threads) against `t10_dual_s0` ckpt 3000 | all pass; equivalence max abs diff 0 (the old chain is reproduced bit for bit, so a speed-up or refactor changed no number) |
| S1 smoke | 20 steps, small set (`--preset smoke --preset small_set`), through run.py `--no-card` | no traceback, loss finite, every module logs `grad.<module>` > 0 except the named unused parts, peak GPU < 11 GB, host RAM < 11.5 GB; s/step noted |
| S2 small run | 400 steps (`--preset small --preset small_set`), eval every 100 | no `nonfinite` / `skipped` / `stalled`; training `total` mean of the last 10 rows < 0.8 x the first 10 rows; no `dead_module` alert beyond the named parts; `val.total` at 400 < at 100; update ratios in [1e-7, 0.2] |
| S3 timing | `t_data` / `t_step` of S2; torch profiler over 8 steps (GPU busy share, the host-side split); inference ms per frame, batch 1, idle GPU, per module | recorded in the report; `data_bound` (> 30 % waiting) is a finding; a step slower than 3.0 s on the small set is a finding |
| S4 probe | every module's output on fixed train and val frames: statistics (finite, std > 0, range), response (zero the module's input: does the output change; `bev_response.py`), maps / boxes / velocity against GT with overlays, the network-part debug views of `bev_suite.py` | each module "makes sense" = finite, not constant, responds to its input, its output is related to its GT where one exists (overlay + number against the null predictor). Every failure is a finding with a cause and a fix |
| S5 fix and repeat | fixes through `commit_paths.py` (a code change of shared code gets an `ar-critic` pass), S1 - S4 again | S2 / S4 gates hold with the fixed code; timing repeated |
| S6 large run | `t16_full.json` unchanged except what S5 fixed (each change listed in the ledger note), 428 / 82 scenes, 6000 steps, run.py with card | health rules below; the metric card of the EMA checkpoint on VAL 1000 frames by `bev_suite.py` |
| S7 control (queued behind S6, `--after`) | the same recipe with the blocks off (`t10_dual_long_s0` config on GT v8), same seed | the matched reference of the baseline; read together with S6 by hand |
| S8 report | videos (predictions against GT, the small set and val scenes), images per module, tables, timing, honest negatives | `check_numbers.py`, then one fresh `ar-critic` in mode numbers |

Health rules of S6 / S7 (`trainsuite watch --kill`, advice first, enforced only for the first two): non-finite loss in 3 rows in a row; `stalled` > 15 min; training `total` rising by > 3 standard errors after step 1500; a module's gradient norm 10 x its own median (`grad_module`). Not a kill rule: `group.motion` flat at 1.0945 (the dead velocity field of every effective-batch-3 run; the hurdle recipe at accumulation 2 had 0.40 at step 4000 on GT v5) is a finding.

## 3 Prediction (guesses; no stored number exists on GT v8 or for this combination)
- S6 against S7, single seed each, VAL 1000, GT v8: the six map IoUs and front mAP2 inside +-0.02 of S7 (the blocks serve the velocity and the radar objects, not the maps); vehicle moving EPE (`vel/dt+0.00/cells/vehicle_moving/epe`) below 4.0 m/s (S7: expected near the dead-field level 6.5); static EPE at most 0.03 m/s above S7; ms per frame at most 25 ms above S7 (t12: M alone 17 ms non-idle).
- Training step time of S6 at most 1.3 x S7.
- Wrong guesses stay in the report.

## 4 Decision rule
- S2 / S4 gates fail -> a finding, a fix, repeat; the large run does not start while one gate is open. A gate that cannot be met by a fix (e.g. the radar memory has no gradient by design) names the block as "off in the large run" with the evidence, and the config of S6 drops it (listed).
- S6 / S7: no KEEP / CUT. The table is the new baseline of the system; comparisons carry "single seed, GT v8". A second seed (twin) is the first item of the module iterations ("afterwards", D44 (4)), because the baseline is only as good as its spread.
- Every number from `metrics.json` / `log.jsonl` by tool, never typed by hand; every table says VAL, 1000 frames, GT v8 (`c643a7539d`); a v5 number next to a v8 number is labelled "not comparable".

## 5 Cost and risks
- Small run: about 15 - 20 min each (time per step is measured in S1, a guess of 2 s). Large run and control: 6000 steps each, 3 - 4 h each at 2 s per step, one GPU line, 10 GB RAM gate.
- Risks: (1) the trainer's GT cache for v8 is cold (cost seconds per sample; the small set is prefilled, the large set takes hours of CPU: `gt_cache_build.py`, started before S6); (2) the radar memory and R + M together are untested; (3) the shared machine (other chats' tests) disturbs timing: timing only on an idle GPU, and the load is written next to every timing number; (4) a speed change of the trainer changes no number only if `equivalence.py` says so.

## 6 Summary (written after the runs)
(open)
