# t19 radar memory - does the Doppler-gated radar memory help the full chain? (card DRAFT 2026-11-02, before any run; written for the lead chat, no GPU used, timing is the lead's call)

Context: `../t16_full_network/experiment.md` (section 6, findings), `../../training/configs/t16_full.json` (`_about`). In the small-set probe of t16 the radar
memory's learned `static` map was exactly 1.0 everywhere and its velocity head got no gradient, so `rad_temporal` is OFF in the large run. The
Doppler static gate already exists in the code (`ModelConfig.rad_static_gate`, `precompute.radar_static`, commit 3c1e680): the memory writes with
`w = min(mass, 1) * static * obs_mass`, and with the gate `obs_mass` is the per-cell mean of `exp(-(vr / 1 m/s)^2)` of the detections (vr
ego-compensated). So the Doppler is the only thing that keeps a mover out of the memory; the learned `static` map stays 1.0 and does not matter for this.

## 1 Card

- **Question.** Does a radar memory (`rad_temporal` "window_memory") with the Doppler static gate (`rad_static_gate` true) lift the full chain
  above the same chain without it?
- **Hypothesis (low prior, stated as such).** Yes, a little: the memory carries the radar of cells it saw before into the present, which helps
  where the camera is weak (far, occluded, bad light) and for objects between two sweeps. The 0.5 s sweep window already gives R a short history
  (age channel), and the t16 probe says the heads answer to R only weakly (maps 17 %, objects 9 % change without detections), so the likely
  outcome is a TIE; the card exists to settle it with numbers and to decide whether the block stays off the ladder.
- **Design.** Arm **A** = the t16 recipe (`t16_full.json`, 6000 steps, batch 3 x accum 2, GT v8, 428 / 82 scenes, R delta slots, M joint, hurdle) +
  `model.rad_temporal` "window_memory" + `model.rad_static_gate` true (new config `t19_radar_memory.json`: configuration only, NO code change).
  Control **C** = `t16.full_s0` itself (rad_temporal "none"; same data, GT version, steps, pinned code commit 40e7ff5b3c = tag `bf/t16.full_s0`).
  Seeds: A_s0, A_s1 and the control's seed twin C_s1 (`t16_full.json` with seed 1) - two seeds per arm (rule of the numbers section).
  Both arms must run under the SAME pinned commit; a later commit changes more than the arm.
- **NOT in the arm: a "prior" (`RadarEncoder.static_weight` as `static_prior` of the memory).** It computes the same quantity as
  `precompute.radar_static`; multiplying it in would count the Doppler twice (`exp(-2 (vr / sigma)^2)`). The wiring patch the lead asked for is
  therefore not needed and none is written. `RadarEncoder.static_weight` (8968380) is a duplicate of precompute's mask; the lead decides whether it is
  removed (one implementation) or kept as the raster-side twin.
- **Metrics** (VAL, `bev_suite.py RUN --velocity eval --n 1000`, same frames and GT signature, `compare` for the paired scene bootstrap):
  headline vehicle AP@2 beyond 40 m inside the fusion area `det/dt+0.00/vehicle/40-60/ap@2` (far range is where a radar memory should help;
  D37 scores the camera-FOV cells); secondary `det/dt+0.00/vehicle/all/ave_skill_moving`, maps (mean IoU of `seg_fov`), front mAP2,
  `seg_obj/seg/dt+-0.50/vehicle/all/all/iou@0`; cost: ms per frame (`bev_suite.py time`, idle GPU) and the module probe of the t16 report (how much the stages change when all
  detections are removed).
- **Prediction** (against C, written before either number exists): vehicle AP@2 40-60 m +0.01 (band +-0.03); `ave_skill_moving` +0.02 (band
  +-0.05); maps and mAP2 within +-0.010; cost +3 GFLOPs (about +2-3 % of the chain; 115 k parameters at 64 channels, `training/README.md`), ms per frame
  +2 to +6 (a guess). In words: a TIE inside every band.
- **Decision rule** (band per metric = max(floor, the 95 % half-width of `compare`); floors: AP 0.03, skill 0.05, maps / mAP2 0.010):
  KEEP (the radar memory with the gate enters the default chain) if the headline is better than C by more than its band on BOTH seeds AND
  no guard is worse by more than its band; TIE (stays OFF, simpler and cheaper) if inside the band; CUT if the headline or two guards are worse by
  more than the band on both seeds.
- **Kill criterion.** Abort an arm run when its in-training validation total at step 2000 is worse than C's at the same step by more than 10 %, or
  when the loss is not finite for 20 steps (the trainer's guard).
- **Cost.** About 6.8 h per run (4.1 s per step, 6000 steps, accum 2): A_s0 + A_s1 + C_s1 = about 20 GPU hours. A cheap rung first: the small set
  (400 steps, 27 min, `--preset small --preset small_set`), to see that the gate lights (the mask is not all 1 or all 0 on the real batch) and that
  nothing breaks; it decides nothing.
- **Inputs.** Ready: the flags (`model.py` `rad_temporal`, `rad_static_gate`; `precompute.radar_static`; the radar cache `radar_hits` /
  `radar_static`), the synthetic gate of section 2, the control run. To build: the config file only.
- **Starts from:** `t16.full_s0`.
- **Referee:** pending (`ar-critic`, mode cards).

## 2 What was measured for this card (no GPU)

`tests/test_radar_memory_ghost.py` (synthetic world of the temporal_fusion self-test: static pattern, radar at the origin, ego standing still, a
dense radar, a blob of 6 m/s; memory read behind the blob compared with the static world; max over frames 10-29):

| case | ghost |
|---|---|
| no gate (obs_mass = field of view only), crossing mover | 0.129 |
| Doppler gate, mover on the line of sight (vr = -6 m/s) | 0.002 |
| Doppler gate, mover crossing the line of sight at x = 16 m | 0.051 |

So the gate removes a ghost completely for a radial mover and about 60 % of it for a crossing mover: the Doppler of a crossing object vanishes at
the crossing point (about 0), which a Doppler-only gate cannot see; a learned velocity (no gradient in the radar branch today) or the vectors of several
radars / sweeps (`tests/test_doppler.py`: tangential speed 1.7-2.4 m/s from one sweep, 0.18-0.57 m/s from six) would have to cover it. On real
nuScenes mini (262,827 detections, 0.5 s window) the Doppler weight of a cell is 0.05 for a detection with |vr| > 3 m/s and 0.97 for one with
|vr| < 0.3 m/s; 18 % of the occupied cells are below 0.5.

## 3 Risks and what the card does not settle

- Crossing movers leave about 40 % of the ghost (section 2); an idealised dense radar was assumed (a real radar is sparse, the gate is 0 where there is no
  detection, so it also stops the memory from filling gaps).
- The learned static map and the radar velocity head stay untrained in this arm (no gradient path); a supervised radar velocity (the radar-corrected flow
  of autogt) would be a later arm, not part of this one.
- The arm adds the 3.0 GFLOPs of the memory at 64 channels and a state to carry between frames (evaluation must use the carried state: `--state carry`).
- One architecture change per arm: the gate is not tested without the memory (the memory without the gate is not an arm: it would write movers).

## 4 Results

(empty - the card has not been run)
