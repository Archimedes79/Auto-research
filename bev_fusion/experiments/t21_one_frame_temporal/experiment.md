# t21 one-frame temporal fusion - a buffer of ONE frame, a pair version and a recurrent version, run as a stream (card 2026-11-03, before any run; Mirko's order D51)

Context: `../../plan/plan.md` section 7.1 (the temporal design: per-branch state, one splat, window of original frames, world memory), `../../training/temporal_fusion.py`, `../../training/model.py` (`cam_buffer`), `../t20_lane_grid_bundle/experiment.md` (the baseline `t20.base_s0` this starts from), `../../records/t21_temporal_2026-11-03.md` (the record of this experiment).

## 0. The card (what the tools run)

```json card
{
 "id": "t21",
 "card": "D51",
 "title": "t21 one-frame temporal fusion: pair (t-1, t) and recurrent state, run as a stream, against the 3-frame window",
 "kind": "block",
 "deterministic": false,
 "metric": "vel/dt+0.00/centre/field/moving/epe",
 "direction": "lower",
 "band": 0.10,
 "prediction": {"value": 1.25, "vs": "c"},
 "arms": {
  "a": {"recipe": "t21.pair_s0: cam_buffer pair, n_cam 2 (frames t-1 and t, 0.167 s apart), warm start from t20.base_s0, 1500 steps, lr 1e-4, ema 0.99"},
  "b": {"recipe": "t21.rec_s0: cam_buffer recurrent, n_cam 3 (the fused state of the previous frame is moved to t and fused with the new frame; carried across keyframes, BPTT through the clip), warm start from t20.base_s0, 1500 steps, lr 1e-4, ema 0.99"},
  "c": {"recipe": "t21.ctl_s0: the window of t20 (cam_buffer window, n_cam 3), the same warm start, steps and lr: the equal-budget control (a gain must not be 'more steps')"},
  "d": {"recipe": "t21.pair4_s0: like a with cam_stride 4 (previous frame 0.333 s earlier = the time baseline of the motion block in t20); the fix candidate for the halved baseline, run after a / b if a is worse than its band. Its stream rate is named next to every number of it: with a buffer 0.333 s old the stream runs at 3 Hz (one new image every 0.333 s) or at 6 Hz with a ring of 2 raw BEVs (not one frame: Mirko's call), so a win is not read as 'one frame at 6 Hz'"}
 },
 "control": "c",
 "variant": "a",
 "rule": [
  {"metric": "vel/dt+0.00/centre/field/moving/epe", "max_loss": 0.10, "note": "by hand: D51, one seed per arm; the same clauses are read for arm b against c (section 1)"},
  {"metric": "detf/dt+0.00/mean/all/mAP", "max_loss": 0.020},
  {"metric": "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@1", "max_loss": 0.010}
 ],
 "budget_gpu_h": 8,
 "kill": "a non-finite loss twice after the fixes, or a version still worse than the control by more than a guard band after two documented fix attempts (then it is dropped with the negative result), or a run that does not finish in 2 x its estimate",
 "evaluate": {"split": "val"}
}
```

## 1. Card

| | |
|---|---|
| lens | Mirko's order D51: a clean temporal mechanism with ONE buffered frame, less calculation |
| group | temporal fusion; cost |
| sources | `plan.md` 7.1 / 7.2, `temporal_fusion.py`, the timing of `t20.base_s0` (lift of 3 frames 28.6 ms, temporal block 13.2 ms, motion block 4.2 ms of 102.1 ms) |
| status | proposed |

- **Question.** Can the camera temporal fusion run with ONE buffered frame instead of a window of three images, as (P) the pair (t-1, t) or (R) a recurrent state that predicts the previous result forward and fuses it with the new frame, without losing quality, and how much calculation does a stream with one new image per step save?
- **Hypothesis.** The window of three lifted frames costs about 19 ms of lifting that a buffer would not recompute, and most of what the older frames give is the velocity (the motion block reads only two frames) and the static memory (world-aligned, already recurrent). P keeps both and drops the third frame: no loss beyond the band. R is the design Mirko asked for ('entirely predicting'); it may lose on moving objects because a chained splat lags (plan 7.1: 0.74 m chained against 0.14 m from the original), but each step has a fresh observation, which that test did not have.
- **Prediction** (VAL, 1000 frames, 82 scenes, GT v8, dt 0, against the control `t21.ctl_s0`, which continues `t20.base_s0` (EPE at moving box centres 1.104 m/s; mAP 0.409; lane divider IoU@1 0.542)): P: velocity EPE at moving box centres 1.25 m/s (the control's 1.10 plus the cost of the motion block's halved time baseline, see the risk below), mAP and lane divider inside their bands; ms per frame in the idle-GPU timing (3 dts, batch path with 2 frames) <= 92 (t20: 102.1); a stream with one new image per step (diagnostic, by hand) <= 80 ms. R: EPE 1.35 m/s (worse: the known drift risk), maps inside the band.
- **Risk named by the motion-block owner (review message of the Temporal Fusion chat, 2026-11-03 23:50).** (1) The motion block M reads the current frame and ONE older frame: with a one-frame buffer at stride 2 its time baseline halves (0.333 s in t12 / t16 / t20, 0.167 s here), the displacement halves and the same position noise doubles the velocity noise; (correction by the owner, 2026-11-04 00:40: M does NOT get dt as an input, dt enters only through its backward warp, so its correction gain is calibrated to the 0.333 s it was trained on: arms a / b start miscalibrated and re-learn the gain during the fine-tune, arm d starts calibrated, so part of any gain of d over a is calibration and not information; and a deployed stream must run at the trained stride, a skipped or late frame biases the speeds; a dt / dt_ref input plane with zero-initialised weights is the code option for later, not tonight). The fix candidates, in this order: arm d (stride 4: the buffer is the frame 0.333 s earlier; the stream then runs at 3 Hz or with a ring of the last 4 frames), or a ring of 2 raw frames for M only (not 'one frame': Mirko's call, so NOT a default). (2) In R, M must see two RAW observations: the fused state that M's own previous field moved is circular. So R keeps the raw t-1 BEV for M (the maps are recurrent, M is pair-fed) and the recurrent state carries only the fused state. (3) Earlier evidence (plan.md 7.4): a carried estimate beat a fresh one by 19-27 % on moving vehicles; a hard start hurt parked cars; a state trained on ONE step drifted badly when streamed (t=1 1.27, t=4 4.63 m/s). So R is trained with the state carried across keyframes (a chain up to 9 frames deep, BPTT), and the drift diagnostic is reported per step t, not as one mean. Fix list for R, first to last: move the fused state with the motion block's velocity field (detached) instead of the temporal block's own velocity head, which reads the fused features it was never trained on and has no direct loss (the supervised field, the one that is scored, is M's; the drift diagnostic reports that head's velocity error at moving box centres on the fused and on the raw features), carry across keyframes (built in), longer clips (seq_len 4-5), the raw previous frame as a second input of the fusion, a sharper kernel in the splat of the fused state, a lower learning rate on the temporal block.
- **Decision rule** (written before the run; one seed per arm, so every reading is labelled "by hand: D51, one seed"): (1) P is adopted as the temporal fusion of the system when no guard is worse than the control by more than its band (velocity EPE at moving box centres 0.10 m/s, mAP 0.020, lane divider IoU@1 0.010) and the cost is lower; inside the bands it is a tie and the cheaper P wins. (2) R is read with the same guards against the control. If R is worse than a band, the lead diagnoses (probe: per-module gradient, the drift of the stream against the unroll, the fused state's mass) and fixes up to twice (candidate fixes: feed the previous RAW frame to the fusion as well, a sharper kernel in the splat of the fused state, carry the state across keyframes with the real step, a lower learning rate on the temporal block); each attempt is a documented run; after two the version is dropped with its negative result (D51: 'rather fix it than not using it'). (3) The stream must reproduce the batch: `tests/test_stream_temporal.py` green and the drift diagnostic (stream over a scene against the unroll at the keyframes) reported, not hidden.
- **Test design.** Three arms, warm-started from the same checkpoint, same steps, lr, seed and split (the control is a continuation with the 3-frame window, so the steps are equal across arms); evaluation by the door, the paired bootstrap of `bev_suite.py compare` against the control for each arm. The comparison to `t20.base_s0` itself is a reference only (2000 more steps). Confounds named: one seed; warm start (the arms differ in what they see for only 2000 steps); the radar branch keeps its own 0.5 s point buffer (out of scope); R trains an unroll of 2 recurrent steps from cold (state not carried across keyframes) and is meant to stream unbounded (the drift diagnostic measures the mismatch).
- **Confounds named by the code critic (2026-11-04 00:45), to be carried into every reading.** (1) Arm b trains at batch 2, accum 3 (the same 6 clips per step) because the chain of 9 recurrent calls needs an estimated 12.7 GB at batch 3 (t20 peaked at 9.67 GB, about 10.9 GB are free next to the desktop); the arms a, c, d keep batch 3, accum 2, so BatchNorm statistics and gradient noise differ a little between b and the others. (2) There is no window arm whose motion block reads the frame just before the current one, so for b the EPE change cannot be split into the buffer effect and the effect of the halved time baseline (d isolates it for the pair only); a window arm with the motion block on frame 1 (a flag, default as today) is arm e, not run tonight unless time is left. (3) b warm-starts the temporal block's relevance, aggregation and velocity head on fused features after BN and ReLU, features they never saw (trained on raw BEV): a loss of b can come from the warm start and not from the recurrence. (4) The pair stream writes the world memory every frame (a 6 Hz stream) while the batch / training path writes it once per keyframe, so stream numbers of the pair are a diagnostic, not a headline; b trains writing at every frame and is consistent with its stream. (5) In the batch path (the door's timing) b runs 3 lifts and 3 block calls, about 26 ms MORE than the window; its saving exists only in the stream (one lift, one block call per new image), so the cost of b and of the pair is stated twice: batch path (door) and stream (`temp/t21/stream_timing.py`, by hand), each with its stream rate. (6) The hop from the previous keyframe's chain to the oldest frame of the next sample uses the gap-mean speed and yaw rate: position error of the carried state against the exact per-sweep chain on 381 mini windows: median 0.04 m, 95th percentile 0.26 m, max 1.06 m. The start line is not `startable.py` on the raw config (it refuses the `split` key like for t20_base) but `python -m trainsuite queue add training/configs/t21_<arm>.json ...` (`temp/d48_t21_queue.sh`), which resolves the split.
- **Cost.** About 1500 steps x 2.4-3.2 s = 1.0-1.3 h per arm, the GPU is shared with the lane job so the arms run one after the other; coding about 2 h (a coder unit, a critic pass); the timing runs need an idle GPU.
- **Kill criterion.** see the card block.
- **Inputs.** Ready: `t20.base_s0` checkpoint (done flag, step 6000), GT v8, the large split. To build: `cam_buffer` ('pair' / 'recurrent') in `model.py` with the stream mode, `tests/test_stream_temporal.py`, three configs `t21_pair.json`, `t21_rec.json`, `t21_ctl.json`.
- **Starts from** `t20.base_s0`.
- **Referee** (to be filled by the critic pass of the code).

## 2. Detail plan

| step | what exactly | where | done when | budget |
|---|---|---|---|---|
| code | `cam_buffer` pair / recurrent, stream mode, state buffer; keep 'window' bit-identical | `training/model.py`, `temporal_fusion.py` | stream == batch to 1e-5; gradient through the carry; cold start finite; window unchanged | 2 h |
| critic | one `ar-critic` pass (mode code) on the diff | - | findings fixed | 0.5 h |
| configs | `t21_pair.json`, `t21_rec.json`, `t21_ctl.json` (new files, `init_from` t20.base_s0) | `training/configs/` | `startable.py` passes, no fresh keys | 0.5 h |
| train | three runs through `run.py` (`trainsuite start`), two slots, pair + rec first, control third | `trainsuite start` | done flags, exit 0 | 5 h |
| evaluate | the door per run (`--all`), timing on an idle GPU; the stream timing and drift scripts | `evaluation/evaluate.py`, scratch scripts | metric cards, tables, timing_modules.json | 2 h |
| decide | the rule above, by hand | ledger note | rows written | 0.5 h |

## 3. Summary (written after the decision)

(after the run)
