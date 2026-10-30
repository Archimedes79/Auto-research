# evaluation - evaluating the BEV system (qualitative + quantitative suite, metrics)

Every number and every picture of a training run comes from here. The machinery is the problem-independent **evaluation suite** `../../autoresearch/evalsuite/` (design: its `ARCHITECTURE.md`, use: its `README.md`); this folder is the BEV *adapter* (the three boxes Source / Subject / Task for a run `runs/<name>`) and the metric library. Run from the `bev_fusion` folder with the `gcar` environment: `wsl ~/miniconda3/envs/gcar/bin/python evaluation/bev_suite.py RUN ...`.

## Use

```
python evaluation/bev_suite.py runs/t1_r4_soa list                                   # the val scenes ("videos") and their frames
python evaluation/bev_suite.py runs/t1_r4_soa qual frame scene-0061/12 --view overview debug     # one frame; overview = camera with the GT, GT, prediction, three error maps
python evaluation/bev_suite.py runs/t1_r4_soa qual debug scene-0061/12              # + the stages of the chain (taps), statistics, plausibility flags, raw taps .npz
python evaluation/bev_suite.py runs/t1_r4_soa qual random 6 --seed 1                # 6 frames: a random video first, then a random frame in it
python evaluation/bev_suite.py runs/t1_r4_soa qual videos 2 --seed 1                # 2 random scenes as H.264 videos (dashboard view, memory carried along)
python evaluation/bev_suite.py runs/t1_r4_soa quant run --n-scenes 20               # numbers over many videos, stored per frame (state carry)
python evaluation/bev_suite.py runs/t1_r4_soa quant outliers eval_n1000             # where the numbers come from: scenes / frames / flags, each with the command that draws it
python evaluation/bev_suite.py runs/t1_r4_soa quant report eval_n1000               # html report with the worst frames drawn
python evaluation/bev_suite.py runs/t1_r4_soa eval --n 1000 [--warm] [--prior 500] [--images 8] [--time]     # the evaluation of a run (see below)
python evaluation/bev_suite.py runs/t1_r4_soa time                                  # ms per frame, idle GPU only -> eval/latency.json
python evaluation/bev_suite.py runs/t1_r4_soa memory --clips 600                    # does the carried memory help? warm vs cold -> eval/memory.json + txt
python evaluation/bev_suite.py compare runs/a runs/b [--tag _warm] [--n 1000]        # paired scene bootstrap b - a
```
Global options: `--device cuda|cpu --workers N --split val|train --ckpt-dir DIR --control shuffle_radar|zero_radar|shuffle_cam|zero_cam --out DIR --state carry|reset|warmup`. Outputs default to `RUN/eval/suite/{qualitative,quant}`. Views of a frame: `overview` (the simple one: grey right, RED false, BLUE missed, cream not evaluated), `outputs` (detailed), `stages` = `debug` (the chain stage by stage), `dashboard` (video frame), `taps` (every tap generically).

`eval` replaces the former `evaluate_run.py`: it runs the suite over the val frames (`--n N` spread over all scenes, cold = every frame from an empty memory; `--warm` the memory carried along whole scenes) and writes the same files as before - `eval/metrics<tag>.json` (flat, same key names), `table<tag>.txt`, `scene_records<tag>.pkl` (for the study tools), `latency.json` - plus the per-frame store `eval/suite/quant/eval<tag>_n<N>/` that everything else (outliers, report, trace, compare) reads. Its numbers equal those of `evaluate_run.py` on the same frames (golden test below).

## Files

| file | content | test |
|---|---|---|
| `bev_suite.py` | the command line of this folder: the suite's commands on a run + `eval`, `time`, `memory`, `compare` and the legacy exports | `bev_selftest.py` |
| `bev_adapter.py` | `build(run, ...) -> (Source, Subject, Task)`: `BEVSource` (val / train keyframes of the run's config grouped into scenes in time order, worker processes read ahead), `BEVSubject` (the run's EMA weights under its own code snapshot `runs/<name>/code`, the chain's state carried frame to frame, forward hooks for ten taps, batch-1 latency), `BEVPriorSubject` (per-cell GT mean as a constant prediction), the controls | contract checker `selftest-adapter` + `--real` |
| `bev_task.py` | `BEVTask`: metrics `seg` (SegMeter counts of the masks fov / seen / out / obj per frame), `det` and `detr` (DetMeter rows: true-positive flags of every detection), `loss`; plausibility checks of a frame | equals the direct meters on every key |
| `bev_views.py` | the pictures as `Page`s: `overview` (camera with the projected GT boxes and map lines), `outputs`, `stages`, `dashboard` | renders; error colours at the right cells |
| `bev_probe.py` | library for one-off scripts that probe a run: pinned code on the path, the chain's functions by name, `load_run`, `load_all`, `run_frame`, `predictions`, `masks`, `gt_boxes`, `add_sample`, `perturb` | static check of all callers |
| `bev_legacy.py` | reader / writer of the pre-suite `scene_records.pkl` and its paired bootstrap, so that runs evaluated before the suite stay comparable (t1, t2 / t3 studies) | equals the suite's compare on the same frames |
| `metrics.py` | the metric library: `SegMeter` (tolerant IoU per layer / range bin / dt, line zero-crossing F1), `DetMeter` (nuScenes-style AP and TP errors; additive per-frame rows `frame_rows` / `reduce_rows`), latency, flat reports | `python evaluation/metrics.py` (known answers; the rows equal the original global matching on random frames) |
| `bev_selftest.py` | known-answer tests of the BEV task on a synthetic BEV world (planted bad frames), `--golden` the numbers of the former `evaluate_run.py` on 24 real frames, `--real` the adapter on a real run | itself |
| `golden/t1_r4_soa_n24_old.json` | the metrics of `evaluate_run.py` on 24 val frames of `t1_r4_soa` (CPU, written before the old code was removed): 3738 numbers | `bev_selftest.py --golden` |
| `evaluate_run.py` | DEPRECATED forwarding stub of the old command line; no script calls it any more (checked 2026-10-30: `temp/study/lane3.sh` calls `bev_suite.py`; `temp/calib/prior_detector.py`, `temp/study/collect.py`, `build_compare_images.py` only name it in docstrings); delete it | `bev_selftest.py` |

The pre-suite scripts (`evaluate_run.py`, `observe_run.py`, `eval_memory.py`, `video_encode.py`) are in `../_to_delete/evaluation_pre_suite/`.

## How the numbers are traced
Every metric stores additive counts per frame (maps: the SegMeter counts; boxes: the true-positive flag of every detection and the GT counts), so any set of frames reduces to the same numbers the direct meters give: all frames, one scene, all but one scene (`quant outliers`: the scenes that drag a headline number most), a bootstrap resample (`compare`). A frame's own scores (drivable IoU, vehicle IoU, recall at 2 m, false alarms, loss) are in its picture's header and in `frames.csv`; `quant trace RUN scene/index` redraws it exactly as the run saw it.
