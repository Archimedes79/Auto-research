# t2_block_swaps - exchange the blocks of the camera + radar BEV chain one by one (card, 2026-10-28)

## 1 Card

**Question.** Which blocks of the chain carry the performance and which are weak? For each block: what happens to the
metrics when it is replaced by a simpler / null block, or by another standard (SOA) building block, all else equal?

**Reference system (REF)** = `t1_r4_soa` architecture: frozen DualBEV image heads -> lift PD -> camera window memory (+ carried
world memory) -> radar raster block -> concat fusion (modality dropout 0.2) -> DualBEV BEV encoder (pre-trained, width 1.0,
stride-1 detail path) -> 96-channel light heads (semantic + CenterPoint object head) at t - 0.5 / t / t + 0.5 s.

**Arms** (ONE exchange each; everything else, data, seed 0, schedule identical):

| block | arm | what changes | kind |
|---|---|---|---|
| (control) | `ref_s0`, `ref_s1` | seed 0 / seed 1 | noise floor |
| lift | `lift_p` | pull without depth weight (Simple-BEV / Fast-BEV) | simpler |
| lift | `lift_s` | forward splat (LSS / BEVPoolV2) | standard |
| camera temporal | `camtemp_none` | current frame only | null |
| camera temporal | `camtemp_gru` | FIERY-style ego warp + ConvGRU | standard |
| radar | `radar_off` | radar branch zero (camera-only system) | null |
| radar | `radar_pillars` | point MLP + scatter-max pillars (PointPillars family; new, `training/blocks_new.py`) | standard |
| camera | `cam_off` | camera BEV zero (radar-only system) | null |
| fusion | `fusion_gate` | GaussianCaR-style rectify gates | standard |
| fusion | `fusion_cmx` | CMX four-stage cross-modal fuser (replaces fusion + trunk) | standard |
| BEV encoder | `trunk_rand` | same architecture, random init | null (pre-training) |
| BEV encoder | `trunk_small` | width 0.5, random init | simpler |
| BEV encoder | `trunk_second` | SECOND + SECONDFPN (CenterPoint / BEVFusion-lidar encoder; new) | standard |

Not exchanged (not possible without raw images in the loop / out of scope here): the frozen DualBEV image heads (cached features),
the object and semantic head designs, the time mover. They are named as untested in the report.

**Protocol.** 3000 optimiser steps x 9 keyframes (3 clips of 3, BPTT, accum 1), lr 3e-4 cosine, flip augmentation 0.5, EMA 0.999,
the 428 training scenes of `t1_r0` ... `t1_r4_soa` (parts 1-6). Evaluation: cold VAL, 1000 keyframes spread over the 82 val scenes
(`evaluate_run.py --n 1000`; same frames for every arm), EMA weights; paired scene bootstrap against REF. About 1.4 h per arm
(measured 1.6 s / step alone), run one after the other (7.4 GB GPU each; two would not fit under 11 GB).

**Hypothesis / predictions** (change of the headline metrics vs REF, cold VAL-1000; mAP = 3-class mAP; written before any arm ran):
- `radar_off`: mAP -0.02 (+/- 0.02), vehicle velocity error AVE worse by > 0.2 m/s, maps about equal.
- `cam_off`: mAP < half of REF, maps drop by more than 0.2 IoU (drivable) - the camera carries nearly everything.
- `trunk_rand`: mAP -0.03, lane / walkway IoU -0.03 (pre-training matters most in a short schedule).
- `lift_p`: -0.015 mAP; `camtemp_none`: -0.01 mAP, in-view maps equal; `fusion_gate`, `radar_pillars`: tie (+/- 0.01);
  `fusion_cmx`: -0.02; `trunk_second`: -0.02; `trunk_small`: -0.01 vs `trunk_rand`.

**Decision rule.** Noise band = max(0.010, 2 x |`ref_s0` - `ref_s1`|) on each headline metric (mAP, drivable IoU, walkway IoU,
lane-divider IoU@0.5 m, vehicle AP@2 m). An arm is WORSE / BETTER when the difference exceeds the band AND the paired bootstrap interval
excludes 0; otherwise TIE. Block verdict: STRONG = the null / simpler exchange is WORSE on a headline metric; WEAK = the null exchange is a
TIE (the block does not carry); a standard exchange that is BETTER is a candidate for the final system; TIE standard exchange = take
the cheaper one (ms per frame, measured on an idle GPU at the end).

**Gate.** Selftests of the new blocks and of every variant (done before the first arm: `training/blocks_new.py`, `training/model.py`).
**Kill criterion.** If `ref_s0` and `ref_s1` differ by more than 0.02 mAP or 0.03 IoU, a third REF seed is added before any verdict;
an arm that diverges (NaN / grad norm > 500 for 100 steps) is cut and reported as such.

**Caveats written in advance.** One seed per arm (the controls give the noise floor only); 3000 steps is about a fifth of the compute
of `t1_r4_soa`, so ranking favours fast-converging blocks (pre-trained, simple) over blocks that need a long schedule; 428 scenes,
front camera; effects found here are effects on this data and schedule, not general laws.

## 4 Phase 2 plan: the ladder (added 2026-10-29 on Mirko's advice: start from a good vision BEV network, add one step at a time, test as you go)

The block swaps above are the subtractive view of one chain; phase 2 reads and extends them as an ADDITIVE ladder with a fallback at every rung.
Rung 0 is the good vision network: pre-trained DualBEV (R50) image backbone + neck + depth head + BEV encoder (all pre-trained, only the lift, the
BEV reduce and the light heads are new) = arm `radar_off`. Every later rung adds ONE thing and is kept only when the decision rule (band, paired
bootstrap) says BETTER, or TIE and cheaper; otherwise the previous rung stays the champion (risk control).

| rung | adds | arms that measure it (this study) | next candidate if the rung is weak |
|---|---|---|---|
| 0 vision BEV | lift + pre-trained encoder + heads | `radar_off`, `lift_p`, `lift_s`, `trunk_*` | longer schedule; unfreeze the last image stage (needs images, not the cache) |
| 1 + radar | radar branch + fusion | `ref_s0` vs `radar_off`; `radar_pillars` | richer radar encoder, radar depth hint into the lift |
| 2 + fusion | concat / gate / CMX | `fusion_gate`, `fusion_cmx` | cross-attention at the stride-16 level |
| 3 + time ("tracking") | window + world memory / GRU | `camtemp_none`, `camtemp_gru` | BEVDet4D-style aligned previous BEV, then an explicit track-association head |
| 4 + data | parts 7-10 (700 train scenes) | final system | more scenes need more cameras unpacked |

Not yet done and needed for rung 0 to count as "good": the vision-only number against the published camera-only level on the same protocol
(the pre-trained DualBEV box head is not loaded here, so only its image/depth part is reproduced - layer 1 gap, named in the report).
