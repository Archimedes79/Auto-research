# t25 one frame, one radar scan, the past recursively at feature level - does the recursive network keep what the 0.5 s radar window and the pair buffer give? (CARD DRAFT 2026-11-14, before any code and before any run; `ar-critic` cards pass pending)

```json card
{
 "id": "t25",
 "card": "t25 one frame one scan",
 "title": "recursive camera (cam_buffer recurrent) + ONE radar sweep per step + recursive radar memory, against the pair-P4 control with 0.5 s of radar",
 "kind": "variant",
 "metric": "detf/dt+0.00/mean/all/mAP",
 "direction": "higher",
 "band": 0.010,
 "prediction": {"value": 0.405, "vs": "ctl"},
 "arms": {"A2": {"recipe": "training/configs/t25_rec_scan_mem.json (to be written): cam_buffer recurrent, radar_window one sweep (about 0.09 s), rad_temporal ego_gru with rad_static_gate, radar recursion per camera step; warm start runs/t20_base_s0, 1500 steps, seed 0"},
          "A1": {"recipe": "training/configs/t25_rec_scan.json (to be written): as A2 without the radar memory (rad_temporal none): what ONE sweep alone costs"},
          "ctl": {"recipe": "training/configs/t21_pair4.json retrained on the current GT (front v13, lane l6, D83): cam_buffer pair, 0.5 s of radar, warm start runs/t20_base_s0, 1500 steps, seed 0 (the stored t21.pair4_s0 was made on GT v9 and its pinned code imports autogt, D81)"}},
 "control": "ctl",
 "rule": [{"metric": "detf/dt+0.00/mean/all/mAP", "min_effect": -0.010}],
 "budget_gpu_h": 6.5,
 "kill": "non-finite loss in more than 1 % of the steps; GPU memory above 11 GB; GT signature of the arms differs from the control's; the stream = batch gate (section 2) fails",
 "evaluate": {"split": "val"}
}
```

## 0 Statement, referee, status
- **Statement.** Mirko (2026-11-13, D91): "Zeitlich möchte ich nur einen Frame als eingang und einen Radarscan, die Vergangenheit soll rekursiv auf Featureebene referenziert werden"; confirmed "Ja, beides" for the two points of the reading: the recursive form replaces P4 as the target, one radar scan = one sweep of the five radars at the camera time.
- **Referee.** The rule in section 1 (point estimates and the paired scene bootstrap of `bev_suite.py compare`, VAL 1000 frames; numbers from the stores, never typed). TEST is not touched.
- **Status (2026-11-14).** Measured: one sweep carries 54 radar points in the grid per keyframe against 300 for 0.5 s; a vehicle has a point in 64 % (82 %), a pedestrian in 29 % (55 %) of the cases (`records/radar_sweep_density_2026-11-13.txt`). Code for the radar recursion is not written (section 2). No run.

## 1 Card
- **Question.** The network gets ONE camera frame and ONE radar sweep per step and carries its past only as feature state. Does that keep the quality of the t21 pair-P4 design with 0.5 s of accumulated radar, at the same training budget?
- **What exists.** Camera: `cam_buffer: "recurrent"` (t21 arm R, a one-seed TIE by hand against the window control: velocity EPE +0.053, the cause is the motion block's halved time baseline, which P4 fixed; the fix list of the t21 card was never tried). Radar: the 0.5 s accumulation at point level; the radar memory (`rad_temporal` with the Doppler static gate) was probed once on the small set and learned nothing there (static map exactly 1.0, velocity head without gradient, because it had no past radar frame): it never got a past to recurse on.
- **Hypothesis (a hypothesis, sign of the radar part unknown).** (H1) The camera part is free: recurrent fusion with the fixed time baseline matches P4 on every row inside the band. (H2) One radar sweep alone (A1) loses detection of pedestrians and far vehicles, where 0.5 s of sweeps give 55 % / 82 % of the objects a point and one sweep 29 % / 64 %. (H3) A radar state recursed per step (A2) recovers most of it: the objects a sweep misses were seen a few steps earlier, and the state moves with the ego motion and the measured Doppler.
- **Prediction (written before the run; VAL 1000, dt 0, seed 0, against the retrained control of the same seed on the same GT).** `detf` mean mAP: A2 at least 0.405 when the control is about 0.415, i.e. within -0.010; A1 a loss of 0.02 to 0.04 (0.375 to 0.395). Velocity EPE of moving objects (`curve.dt+0.00/velocity/moving/epe`, lower is better, control about 1.41 on GT v9): A2 at most +0.10, A1 +0.2 or more. Lane divider IoU@0.5 and the drivable area do not move beyond the band (the camera carries them). A2 - A1 at least +0.010 on `detf` mAP.
- **Design.** Three arms, one training at a time, the same schedule: warm start `runs/t20_base_s0` (EMA), 1500 steps, accumulation as in `t21_pair4.json`, GT v13 / l6. A1 isolates what one sweep costs, A2 what the radar recursion gives back, ctl is the matched control. A radar change from 0.5 s to one sweep makes the warm start's radar encoder see 5.6 times fewer points: the first 200 steps are expected to lose; the schedule is not lengthened (an equal-budget comparison).
- **Decision rule (fixed now, before any run; d = arm minus control from `compare`, `detf` mean mAP, interval = paired scene bootstrap).** (1) **A2 holds** when d >= -0.010 and the interval's lower end is above -0.015, velocity EPE of moving objects <= +0.10, lane divider IoU@0.5 not below -0.010: the recursive one-frame, one-scan network is the new target design; next a from-scratch base with it. (2) **A2 fails** when d < -0.020 with the interval's upper end below -0.010: then the A2 - A1 gap says whether the radar memory helps at all; the fix list is the write mask, a second sweep as the input (asks Mirko), the step rate. (3) Everything between, or A2 - A1 < +0.010: **INCONCLUSIVE**, a second seed only on Mirko's word. A1 only explains; it decides nothing.
- **Noise.** `detf` mean mAP of P4 against the control in t21: +0.0037 [-0.0014, +0.0091] (paired); a fresh control pair of equal recipes sets the real floor (the rule's band 0.010 is the card's, the control seed pair is not run here: 2 seeds only on Mirko's word).
- **What it does not test.** From scratch training; a different step rate (6 Hz as in t21, not 12 Hz); a two-sweep input; the object and map heads' own changes; speed (ms per frame is measured, not ruled on).
- **Cost.** Code about a day of lead work (section 2) and CPU tests; GPU per arm about 1.9 h training + 0.2 h evaluation, three arms about 6.5 h, one training at a time (D57: Spurmitte has the GPU first; windows after its chains). The GT stores of v13 / l6 for BEV are built (2026-11-12).

## 2 Detail plan (the coding step; for `ar-coder`, with a critic pass of the shared code)
1. **Dataset** (`training/nuscenes_dataset.py`): option `radar_per_frame` (default False, nothing changes): `item["radar_seq"]`, one radar dict per camera frame of the window (index 0 = the current frame), each holding the newest sweep of each of the five radars not after THAT frame's camera time (age <= `radar_sweep_age`, 0.09 s), moved into the ego frame of that frame; `collate` stacks them with the batch index; `batch["radar"]` stays the current frame's. Test: the points of `radar_seq[0]` equal the points of the `radar_window=0.09` radar; mean count about 54 per keyframe on mini.
2. **Model** (`training/model.py`): `_unroll` runs the radar branch per frame next to the camera branch when `rad_temporal` is set and the batch holds `radar_seq`: encode the frame's sweep (radar block), step the radar temporal block with past = [(previous fused radar BEV, ego chain)] like the camera, state `state["rad"]` and `buf["rad_fused"]`; the write mask of the memory per frame (`radar_hits` / `radar_static` of that frame's sweep). `forward` fuses the newest frame's camera and radar outputs as today; the side loss reads the newest frame only. `advance` (stream) takes the same path for a frame that needs no output, so a stream of one frame and one sweep per step is possible (today it refuses `rad_temporal`). Default configs: bit-identical outputs.
3. **Tests** (`tests/`): (a) stream = batch: a clip of 6 frames through the batch path and through one-frame stream steps gives the same output and state (tolerance as `test_stream_temporal.py`); (b) gradient: the radar block, the radar memory's velocity head and the write gate get a gradient in A2; (c) default path unchanged (existing configs, outputs equal to the stored selftest values); (d) `check_config` refuses `radar_per_frame` without `cam_buffer: "recurrent"`.
4. **Configs** `t25_rec_scan.json`, `t25_rec_scan_mem.json`, `t25_ctl.json` (from `t21_pair4.json`, name and `_about` and the keys above only; pinned by a test as `tests/test_lane_pull.py` does for t24).
5. **Order.** Dataset and test, model and tests, configs; then the smoke (a few steps on the small set), then the card goes to `ar-critic` (mode cards); no GPU hour before both.
