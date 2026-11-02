# t16 full network - does the complete chain (every built block, five-layer framework, GT v8) train, and what is its baseline? (bring-up card, 2026-11-02, before any run)

## 0 Statement, referee, status
- **Order (D44, Mirko 2026-11-02, word for word in `experiment_execution/DECISIONS.md`):** generate a full network with fusion; train it first on a small set; time the training code; probe the outputs of all modules ("does it make sense?"); find errors and fix them; train again until fixed, with timing; then train on the large set and establish a performance baseline; make a report with result videos and images; afterwards module improvements and the repeat of old experiments on the new system and GT v8.
- **What this card is.** A bring-up with gates, not a hypothesis test. The small runs are diagnostics (`--no-card`, they feed no verdict). The one run that counts is the large run: it becomes the baseline row of the new system. Everything it is compared with is labelled by hand ("by hand: single seed, bring-up baseline, no verdict"), because one seed never gives KEEP or CUT (project rule).
- **Status (2026-11-02, updated after the small-set bring-up).** Sections 1 - 5 are the plan as written before any run; section 6 is the log of the small-set stages S0 - S4 (runs, timing, probe, findings, what the large run changes); the large run and its control are queued behind the final small run. Main was pushed (b430af0) after every other chat had committed (D44 (2)).
- **TEST is untouched** (VAL only; hard rule 1).

## 1 The network (one config, every block that is built and has a passing gate)
| slot | kind in `t16_full.json` | source / gate |
|---|---|---|
| camera encoder | `camera_features` "dual": cached DualBEV heads (stride 16) + BEVFusion map-trained features (stride 8), two lifts PD merged by a 1x1 conv, 3 frames | t10 chain; `test_dual_camera_geometry.py` |
| camera temporal | `window_memory` (window of the last frames + world-aligned static memory) | t2 / t10 |
| radar encoder | `delta_slots` R, `c_rad` 64, side loss `radar_aux` 1.0 -> 0.05 | Radar Encoder chat, `test_radar_encoder.py`, `test_radar_slot.py` |
| radar temporal | `window_memory` with the Doppler static gate (`rad_static_gate`): **in the small-set runs, OFF in the large run** (section 6, finding 3) | built, never run before t16; its window part and velocity head get no gradient without past radar frames (named in `test_slots.py` KNOWN_UNUSED); after 400 steps its static map is 1.0 everywhere |
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

## 6 Bring-up log (2026-11-02, small set; numbers from the stored files of the runs, single seeds, VAL = the 2 val scenes of autogt's small set, 77 frames, GT v8 `c643a7539d`)

**Runs** (all through `run.py`, `--no-card`, diagnostics, no verdict): `t16.smoke_s0` (20 steps), `t16.small_s0` (400 steps, the full config of section 1 with the radar memory), `t16.small_ctl_s0` (the matched control, `t16_ctl.json`), `t16.small_bg1_s0` (`loss.motion_w_background` 1.0), `t16.small_clip_s0` (`grad_clip_split` ["motion_block"]). Evaluations by `bev_suite.py RUN --curves --fusion --velocity eval --n 80 --prior 300`; the module probe and the timing by the scratch scripts `temp/t16/probe_modules.py` / `time_modules.py` (promoted into `evaluation/` at their second use, the large run).

**Gates.** S0: `test_slots.py` (66 s, 13 combinations incl. the full one) passes; `run_gates.sh`: everything passes except `velocity hurdle`, a stale assertion (LOGGED_TERMS grew by the two radar side-loss terms that `train.forward_loss` makes, the toy BEVLoss forward has none): fixed in b430af0, the gate passes. S1: 20 steps, loss 33.6 -> 31.4 (val 26.2 -> 23.5), finite, a gradient on all 14 modules, peak GPU 10.1 GB, 3.7 s per step. S2: 400 steps, no alert of the trainsuite, val total 20.2 (step 100) -> 18.2 (step 400), 4.1 s per step of which 7 % waiting for the data, update ratios 1e-4 - 5e-3.

**Timing (S3, idle GPU, batch 1, dt -0.5 / 0 / +0.5, the cached chain without the frozen image heads; `eval/timing_modules.json` of both runs).** Full network 102.8 ms per frame, control 86.2 ms: +16.6 ms (+19 %). Per module, full / control: lift 14.1 / 14.0, camera temporal 13.1 / 13.8, trunk 11.0 / 10.8, motion block 4.5 / -, radar memory 4.2 / -, radar encoder 4.0 / 1.9 (delta slots against the plain raster); the rest (time mover, heads, decoding: not hooked) 39 / 33 ms. One training sample, forward + loss / backward: 161 / 95 ms against 140 / 92 ms; peak 1.6 GB per sample.

**Matched control on the small set (400 steps, 8 train scenes; one seed each: a diagnostic, not a verdict; the seed spread is not measured here).**

| metric (VAL, 77 frames) | full | control |
|---|---|---|
| drivable IoU@0 / lane divider IoU@0.5 | 0.597 / 0.100 | 0.660 / 0.101 |
| walkway / carpark / crossing ROC-AUC | 0.819 / 0.677 / 0.691 | 0.849 / 0.809 / 0.521 |
| mAP (fusion area) / vehicle AP@2 / pedestrian AP@2 | 0.228 / 0.630 / 0.222 | 0.207 / 0.584 / 0.322 |
| velocity EPE at moving box centres (zero predictor 2.60) | 1.40 | 1.92 |
| velocity EPE, vehicle_moving cells / static cells | 1.76 / 1.26 | 2.93 / 0.51 |
| val group.motion | 0.598 | 0.368 |

The full network is better on moving objects and worse on the static background; the maps are inside what two scenes and one seed can tell (drivable -0.06, the rest mixed).

**Probe of the modules (S4; `runs/t16_small_s0/eval/modules/`: `frame_<k>.png`, `response.json`, `static_check.json`).** Every module's output is finite, not constant, without a dead channel; the camera BEV fills the camera cone and sits on the vehicles, the radar points are where the detections are, the fusion and trunk features carry both, the maps recover the road corridor (the prior baseline's drivable IoU is 0.42) but not yet the cross road, lane dividers 0.10 against the prior's 0. Response when an input is removed (relative change of the stage output, 3 cold frames): without the camera the camera BEV 0.97, fusion 0.70, maps 0.43, the velocity field 1.07; without all radar detections the radar raster 1.0, fusion 0.59, maps 0.17, objects 0.09, velocity 0.24; without the Doppler alone the velocity 0.16 and the motion block 0.2 (the Doppler is used). The box velocity of the object head equals the motion block's field (`box_vel_from_field` works). Warm frames (the memories read what the first call wrote): the camera memory reads (mem_read std 0.72), the radar memory writes only where detections are (86 % zeros).

**Findings** (what the bring-up found; none stopped training):
1. **The motion block's gradient dominates the clip.** `grad.motion_block` 29 - 133 (median 80) against 0.1 - 41 for the other modules; the global clip (5) scales every update by 0.04 - 0.14. The Temporal Fusion chat: expected (t12 saw 117 -> 22), AdamW is insensitive to a constant factor, not to a fluctuating one. Built: `TrainConfig.grad_clip_split` (opt-in, off = bit-identical; code critic REVISE -> all findings fixed, e1e28f2). The critic's measurement on the probe rows: the rest of the network stays clipped every step (factors 0.08 - 0.29), the split raises its level about 3 x and leaves the step-to-step spread almost unchanged (max/min 3.6 against 3.9): the gain is small, and the control chain runs in the same regime (norms 12 - 35). The split is used in the large run for that reason; its small arm (`t16.small_clip_s0`) checks that it does no harm, it does not prove a gain.
2. **Static cells of the motion block: 1.26 m/s against the control's 0.51.** The mean field on static cells is about 0 (|mean| <= 0.37 m/s at ego speeds of 2 - 5 m/s, std 0.6 - 1.6 m/s): zero-mean noise, not an ego-motion offset or a frame error (`static_check.json`). The block starts at exactly zero and its output is scaled 10 x per level, so it moves ten times faster per step than a 1x1 head; in the t12 log it settled only after about 2000 steps. The weight of the background term (0.1) is not the lever: with 1.0 (`t16.small_bg1_s0`) the static EPE is 1.04 against 1.26 and everything else is the same within the noise. Not changed; watched in the large run (`vel` rows at every evaluation).
3. **The radar memory (rad_temporal + rad_static_gate) is an unlearned block.** After 400 steps its learned static map is exactly 1.0 everywhere (std 0), its velocity head and window part get no gradient (the known KNOWN_UNUSED of `test_slots.py`); the camera memory's static map is 0.9997 +- 0.009. The Radar Encoder chat advises leaving it out (it writes moving content into the memory; the 0.5 s sweep history is already in the raster). **Off in the large run**; back as its own arm when the gate has a signal; 4.2 ms per frame saved.
4. **`radar_kept` falls 0.94 -> 0.04 in 400 steps** while the side loss decays (half-life 0.2 of the run = 80 steps). The Radar chat: expected in a short schedule (its 1500-step run: 0.34, 0.69, 0.71 at steps 500, 1000, 1500; the monitor is strict); not a defect, checked again in the large run.
5. The hurdle's moving probability is high almost everywhere after 400 steps (it is supervised at box centres only).
6. The evaluation's debug taps (`bev_adapter.bev_taps`) do not cover the new modules (delta slots, radar memory, the motion block's levels, the hurdle); the probe script does; to be integrated into `evaluation/` after the large run.
7. Walkway / carpark / crossing IoU is 0.0 at 400 steps (the prior's too) while the curves say the heads carry information (walkway AUC 0.82): rare classes are not over the 0.5 threshold yet, a training-length effect.
8. The trainer's GT cache for v8 costs 0.57 s per sample per worker when cold (the small set: 37 s for 230 samples with 4 workers): no prefill is needed for the large run (6 loader workers give about 10 samples per second against 2.3 needed).

**Changes for the large run (S5):** the radar memory off (finding 3), `grad_clip_split` ["motion_block"] on (finding 1), everything else as section 1; the small run of this final config (`t16.small_final_s0`) repeats S2 / S3 before the large run starts.

## 7 Results (written 2026-11-02, updated as runs finish; every number from `runs/<run>/eval/by_n/table_n1000.txt`, never typed by hand)

**Large baseline `t16.full_s0`** (config `t16_full.json`, 6000 steps, 428 train / 82 val scenes, GT v8, started 04:47, finished 11:53, 3.61 s per step, no trainsuite alert; code commit `40e7ff5b3c`, tag `bf/t16.full_s0`).
Validation total 11.25 / 8.27 / 7.89 at steps 2000 / 4000 / 6000; the radar-kept monitor is back at 0.90 (it fell to 0.04 in the 400-step run); the train-minus-val gap of the six map IoUs is 0.091
(drivable 0.040, walkway 0.036, carpark 0.171, crossing 0.072, lane divider 0.125, road divider 0.100; `bev_suite.py generalisation --n 1000`).

| metric (VAL, 1000 frames, GT v8, one seed) | t16.full_s0 (baseline) |
|---|---|
| drivable IoU@0 | 0.779 |
| walkway IoU@0 | 0.413 |
| carpark IoU@0 | 0.274 |
| ped crossing IoU@0 | 0.256 |
| lane divider IoU@0.5 | 0.429 |
| road divider IoU@1 | 0.572 |
| mAP, fusion area (detf) | 0.416 |
| vehicle AP@2 | 0.799 |
| pedestrian AP@2 | 0.376 |
| velocity EPE at moving box centres (m/s) | 1.112 |
| zero predictor at the same centres | 4.268 |
| EPE vehicle_moving cells | 1.441 |
| EPE static cells | 0.189 |

The motion block's error on static cells settled from 1.26 m/s (400-step run) to the value above: the 400-step number was early-training noise (finding 2). The prediction of section 3 (baseline against control within +-0.02 on the maps,
moving EPE below 4.0 m/s, static EPE at most 0.03 above the control) is judged by hand when the control has finished ("by hand: one seed per arm", no KEEP / CUT).

*Queued behind it (all through the launcher, pinned code):* the extraction of the BEVFusion features for 272 more train scenes, `t17.data` (700 train scenes), `t18.fix` (t17 recipe + the Training chat's hygiene flags + the motion block's own learning rate), the matched control `t16.ctl_s0`, each with an evaluation job.
