# What the first observability tool (`evaluation/observe_run.py`, `video_encode.py`) learned that the evalsuite should keep

Written 2026-10-30 by the main BEV session after a day of looking at real pictures of t1_r4_soa. The suite core is already close to
these tools (energy / PCA, TapRecorder, H.264 encoder, sampling video-then-frame); this list is only what the real runs taught.

## A. Faults that would be repeated if not designed out
| # | what happened | what the suite should do |
|---|---|---|
| 1 | **The read-back check of the video cannot see the codec**: OpenCV decodes MPEG-4 part 2 ('FMP4') as happily as H.264, so "frame count + first frame not black" passed on the file that did not play in the browser | `verify()` also reads `CAP_PROP_FOURCC` and requires `h264` / `avc1` (done in `bev_fusion/evaluation/video_encode.py`; the suite's `render/video.py` has the old check) |
| 2 | a "difference between two stages" panel (fused - current frame) is meaningless when the stages live in different feature spaces; it looked plausible in every frame | rule for TapSpecs: never subtract tensors of different stages; show what a block *supplies* (its aux outputs) or do an ablation (rerun the block without the input) |
| 3 | a config without scene lists meant 428 val scenes on 2026-10-26 and 150 today (new data parts), and 46 % of the val samples had no cached camera features (KeyError in the loader) | `Source.scenes()` must list only frames whose inputs exist and print the coverage; a Source for the BEV runs takes the frozen lists `bev_fusion/data/scenes_parts1_6.json` when the config has none (`observe_run._freeze_scenes`) |
| 4 | boxes counted against all GT of the grid instead of the front field of view that `DetMeter` scores: "2 found / 26 GT" looked like a disaster | the Task's checks must use the same filters as its Metric (reuse `in_front`, observed-only) |
| 5 | the viewer threshold (0.3) was a guess | derive it from data: best-F1 threshold on the sampled frames (`score_probe.py`: 0.30, precision 0.81, recall 0.57 for vehicles) and print the precision-recall at a few thresholds next to AP |

## B. Taps that need more than "one tensor per hook" (BEV subject)
- `cam_temporal` returns `(out, state, aux_dict)`; the useful pictures are in the aux (`agg` = window frames moved to now, `mem_conf` =
  where the world memory is confident). `Hook.pick` returns one value: allow `pick` to return a dict -> several taps from one hook, or
  one Hook per aux key.
- `lift` and `cam_reduce` run once per sample and frame (`select="list"` is right), their frame 0 is the current one.
- the camera input is not a module input in the cached model (`batch["prob"]`, `batch["ctx"]`): the Subject adds those arrays itself
  (expected depth, entropy); the model's own image warp `ida` is needed to draw the crop the network saw (`camera_image`).
- sequence-trained runs need `warmup = seq_len - 1` previous keyframes through the model with the carried state (`run_frame`); the
  video carries the state over the whole scene (38 keyframes, training used 3): keep both modes (`carry` / `reset`).
- decode at dt != 0 with the TRUE gap and the gap-mean ego motion (`train.forward_loss` does it with `cfg.gap_ego`): a Subject that
  calls `model(batch, dts=...)` directly decodes at the nominal 0.5 s and mixes in 0.2-0.35 m of label error.

## C. Checks worth porting (names and the thresholds used; all heuristics)
no NaN / dead channels per stage; camera BEV energy outside the camera cone < 5 %; median expected depth 4-40 m; radar detections >
20 per window; moved-past-frames energy / current frame in 0.05-3; memory confidence share; drivable share prediction vs GT within
0.15; boxes found >= 50 % of GT in the front view and detections <= 2 x found + 2; splat known cells 30-99.9 %; **velocity head:
predicted / GT speed on MOVING vehicle cells > 0.5 (warn > 0.2, bad below)**.

## D. Numbers that the metrics must report because the plain mean hides them
- **velocity**: AVE 1.82 m/s over all matched vehicles looks fine, but the moving ones (20 % of vehicles) have error 5.44 m/s against 7.60 for
  the zero predictor and the dense head is 17 % better than zero. Report velocity error split moving / static, the zero predictor next
  to it, and the share of moving cells predicted below 0.5 m/s (`velocity_probe.py`).
- every learned quantity gets its **null predictor** in the same table (zero velocity, prior map, constant): the BEV runs had no such row
  for the velocity head until a picture showed it.
- **information oracles as data-only checks** (no network): median radar Doppler inside a moving GT vehicle box vs its GT radial speed
  (corr 0.989, error 0.45 m/s, 94 % of moving vehicles have detections; `doppler_info.py`). It says which input could carry the signal
  before anyone blames the network.

## E. Panel texts
The guide table of `observe_run.GUIDE` (what each panel should look like, what a red flag is, 12 + 12 panels) is ready-made `TapSpec.note` /
`redflag` text.
