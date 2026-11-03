# trainsuite - start, supervise and stop a training from the files it writes

Who does what in a training (the folders are the BEV system's; another project plugs in the same way):

| part | where | job |
|---|---|---|
| model, loss, data | `bev_fusion/training/` (`model.py`, `losses.py`, `nuscenes_dataset.py`, `train.py`) | what is trained; `train.py` builds them and calls the engine |
| the engine | `autoresearch/trainkit/` (contract model / loss / data / config: [README](../trainkit/README.md)) | the loop: accumulation, EMA, clipping, checkpoints with resume, skipping non-finite steps, the stop request |
| the launcher | `autoresearch/tools/run.py` | card, ledger row, code tag, GPU slot, retry |
| this suite | `autoresearch/trainsuite/` | compose a config, start, queue, status / health / watch / curves, stop; no project code, standard library only (`curves` also needs matplotlib) |

Run every command from `autoresearch/`. `AR_PROJECT=../bev_fusion` names the problem (the default is the lane one); `R=../bev_fusion/runs` is the folder of the run directories. `python -m trainsuite selftest` checks the suite on synthetic runs in seconds.
```
export AR_PROJECT=../bev_fusion; R=../bev_fusion/runs
```

## 1. What a config and a manifest look like
A training is one JSON file with a value for every choice: `bev_fusion/training/configs/t20_base.json` (an unknown key is an error). The keys that matter:

| key | meaning |
|---|---|
| `name`, `out_root` | the run folder `runs/<name>`; `start` sets `name` from `--id` |
| `split` | `"large"`: the scene lists `bev_fusion/splits/large.json` (`small` 60 / 10 scenes, `large` 428 / 82, `xl` 700 / 82) |
| `steps`, `lr`, `warmup`, `batch_size`, `accum`, `seed` | the schedule; effective batch = `batch_size` x `accum` (3 x 2) |
| `feature_cache`, `feature_cache2`, `gt_cache`, `radar_cache`, `lane_images` | the derived data the run reads (caches, not in git) |
| `model`, `loss` | `ModelConfig` (which blocks: `camera_features: "dual"`, `lane_branch`, `motion_block`, ...) and `LossConfig` (head layout, loss terms); the layout flags (`centreline_1m`, `radar_stationary`) must agree in both |
| `log_every`, `eval_every`, `save_every`, `keep`, `probe` | what is logged and saved, how often (section 6) |
| `_about` | a comment (keys starting with `_` are ignored): why the file exists |

Variants are values, not code. A **preset** is a JSON of overrides (`bev_fusion/training/configs/presets/`: `small` = 400 steps, `smoke` = 20 steps, `onfly` = no caches); `--set key=value` changes one value (`a.b=c` for nested keys, `c` is read as JSON). `start` (section 3) prints every change with its old value and marks a key the base did not have as NEW, so a typo shows.

What a run records, next to its checkpoints:
- `runs/<id>/config.json`: the config the trainer ran with. `start` replaced `split` by the exact scene lists, so the run is pinned to the lists, not to the split file.
- `runs/<id>/gt.json`: the ground-truth signature and version it trained on.
- `runs/<id>/manifest.json`: the identity in one small file. It is made after the launch (the commit exists only then), from `bev_fusion/`: `python training/splits.py manifest runs/<id> --card records/runs/<id>/run.json`. For `t16_full_s0` it holds
```
{"run": "t16_full_s0",
 "split": {"name": "large", "train": {"n": 428, "hash": "b694b2e11f43"}, "val": {"n": 82, "hash": "e290142492cd"}},
 "gt": {"signature": "0.5_4_6a7aa7864a", "version": "v8"},
 "code": {"commit": "40e7ff5b3c", "tag": "bf/t16.full_s0"},
 "card": {"hypothesis": "...", "prediction": "...", "rule": "..."}}
```
- the card (hypothesis, prediction, decision rule) is given at the start (`--hyp --pred --rule`) and kept in `bev_fusion/records/runs/<id>/run.json` and the ledger row; the experiment's story is `experiments/<id>/experiment.md` (open) or `results/experiments/<id>/` (finished).

## 2. How a run is tied to its data and its code
| what | pinned by | not pinned |
|---|---|---|
| scenes | `split` becomes the lists in `runs/<id>/config.json` (name, counts and hashes in `manifest.json`); a new split is a new file in `bev_fusion/splits/` ([README](../../bev_fusion/splits/README.md)) | |
| ground truth | `gt` in the split file = the autogt version its caches were checked against (v8); the run writes `gt.json`; evaluation compares only runs with the same signature | `autogt/` is imported live; the run's tag records its state |
| derived data | the cache paths in the config; `bev_fusion/data/README.md` says how each is built; `start` runs the project's `prepare` check (is the GT store complete for these scenes?), `--prepare` builds a missing one as a run of its own | the caches, raw nuScenes and the pretrained weights are files, not versions |
| experiment code | `run.py` commits the code and tags it `bf/<id>`; `run_ar.sh` unpacks `bev_fusion/training` and `autoresearch/trainkit` of that commit into `runs/<id>/code/` and trains from there | editing `training/` while a run trains changes nothing for it; a resume trains from a new snapshot |
```
cat ../bev_fusion/runs/t16_full_s0/code/SNAPSHOT.txt      # commit 40e7ff5b3c (git archive of bev_fusion/training; run t16.full_s0, tag bf/t16.full_s0)
git tag -l 'bf/t16.full*'                                 # bf/t16.full_s0, bf/t16.full_s0.eval
```

## 3. How to start
```
python -m trainsuite start training/configs/t20_base.json --id t20.demo --preset small --set split=small --set steps=600 --no-card
```
Without `--go` this is a preview: nothing starts, the composed config goes to `bev_fusion/temp/compose/<id>.preview.json` (scratch). It prints the changes, the notes of the split resolution, the project's static check and `prepare` check (`training/startable.py`, `gt_scenes.py`), and the exact `run.py` command:
```
t20_base.json -> temp/compose/t20_demo.json: 10 change(s)
  steps: 6000 -> 400  (preset small)  ...  split: "large" -> "small"  (--set)   steps: 400 -> 600  (--set)   name: "t20_base" -> "t20_demo"  (--id)
  note: resolve: train_scenes = 60 entries     note: resolve: val_scenes = 10 entries
cd <bev_fusion> && AR_PROJECT=<bev_fusion> python3 .../tools/run.py --id t20.demo --gpu --split train --no-card -- bash temp/train/run_ar.sh temp/compose/t20_demo.json
(nothing started: add --go)
```
- **`--go`** starts it detached and waits 3 s to see whether `run.py` refused the id (held, running, finished, no card). **The same command again resumes** a stopped or crashed run from its last checkpoint.
- **The card.** A run that may feed a verdict needs `--hyp "..." --pred "..." --rule "..."`. `--no-card` is for diagnostic runs (smoke, bring-up, timing): the ledger row says "no card" and the run never feeds a verdict.
- **`run.py` itself** (`python3 tools/run.py --id ... --gpu -- <command>`) is what `start` calls; its options (`--after`, `--timeout`, `--retries`, `--ram-mb`) are in its docstring, the BEV start line in `bev_fusion/records/README.md`.
- **GPU slots.** `--gpu` waits for one of 2 slots (shared with the lane project, one GPU) and for 10 GB of free RAM; section 7.

## 4. How to stop and resume
```
python -m trainsuite --root $R stop t20_demo --reason "why"
# stop requested for t20_demo: .../runs/t20_demo/CONTROL
# the trainer sees it at the next step boundary, saves a checkpoint, writes STOPPED and exits with code 75
```
- At every step the trainer looks for `CONTROL`. It saves a checkpoint, writes `STOPPED` and exits with **75**, never 0, so no `.done` exists and nobody evaluates half a model.
- `run.py` calls exit 75 `stopped` (not a failure, no retry) and puts the id **on hold**: it refuses the id and blocks its `--after` dependents until the person's words clear it: `python3 tools/hold.py clear <id> --words "<their words>"` (`hold.py show` lists the holds). Then the same start command resumes the run, from a new snapshot of the code: stop, fix, start again is the hot-fix loop.
- A crash (traceback, OOM, CUDA error) has no hold: start the same command again; `run.py` retries a CUDA "unknown error" by itself.
- `trainsuite resume <run>` moves a stale `CONTROL` / `STOPPED` aside by hand (the trainer does it itself at its start). A snapshot older than this protocol ignores `CONTROL`: `stop` refuses it (exit 2).

## 5. How to supervise
| command | answers |
|---|---|
| `status [RUN]` | where is every run (state, step, ETA, loss); for one run also the loss groups, the module table, the time split, the last eval |
| `health RUN [--vs REF] [--kill RULE]` | is it sound: crash, stall, NaN, spike, rising loss, dead module, data-bound; exit 1 on an alert or a kill hit |
| `watch RUNS [--enforce]` | one line per change until all are finished, made for an agent's Monitor; `--enforce` stops a run on a kill hit |
| `curves RUN [RUN]` | a png of the loss, its groups and the gradient norm; needs matplotlib (the `gcar` python) |
| `speed BASE FAST` | s per step, speed-up, data wait, GPU busy, peak memory, loss at the last shared step |
| `early RUN --vs CONTROL` | advice from the gap to a control run, calibrated on finished pairs; stops nothing |
| the **module table** (in `status RUN`) | per module the gradient norm and the update size per step; gradient 0.0 = that module learns nothing (needs `probe`) |
| **kill rules** | the card's kill criterion as a line: `--kill "group.lines@1000 > 1.48"` or `<run>/kill.txt` |
```
python -m trainsuite --root $R status                                  # run  state  step/total  ready  s/step  ETA  last step  loss  flags
python -m trainsuite --root $R health t16_full_s0 --kill "group.lines@1000 > 1.48"
#   [kill ] group.lines@1000 > 1.48: ok (value 0.6306)    modules (median of the last 20 rows): trunk  grad 2.12  update 6.18e-06 ...
python -m trainsuite --root $R watch <run> --enforce --kill "group.lines@1000 > 1.48"      # runs until <run> is finished: start it as a Monitor
#   [16:21:04] t16_full_s0: kill rule 'group.lines@1000 > 1.48': ok (value 0.6306)       (what `watch t16_full_s0 --once ...` prints for a finished run)
```
A healthy run is not a good network: the checks say the training is sound, the evaluation says whether the result is good. All checks, their thresholds, the stop details and the kill-rule syntax: [EXTENSIONS.md](EXTENSIONS.md) chapters 2, 3, 8, 9, 10.

## 6. How to log more for deep debugging
A row of `log.jsonl` has the loss, its groups, `grad_norm`, `lr`, `s_per_step`, `gpu_gb`. Switch on more with config values (`--set key=value`):

| value | adds | costs |
|---|---|---|
| `probe: true` (on in `t20_base`) | per module `grad.<module>` and `upd.<module>` (gradient norm, update-to-weight ratio), `t_data` / `t_step` (seconds waiting for data / the whole step) in every `log_every` row; `incident_<step>.json` (step, kind, sample ids of the batch, losses; at most 50) when a step is skipped as non-finite | one copy of the trainable weights per logged step |
| `grad_diag_every: N` | every N steps the cosines of the task-group gradients (do the tasks pull against each other?); not under fp16 | extra gradient work on those steps (not measured) |
| `profile_steps: n` (n <= 8) | `profile.txt` in the run folder: torch.profiler over n steps after 10 warm-up steps (GPU busy share, host synchronisations per step, top operators) | host RAM (30 steps held 22 GB): a short run on a quiet box only |

Without a run: `bev_fusion/training/debug.py` needs one batch, no GPU: `overfit_one_batch` (the loss must go to about 0; if not, the bug is in the loss, the targets or the wiring), `frozen_stayed`, `anomaly()` (names the operation of the first NaN), `replay_ids` (rebuilds the batch of an incident). `probe.py` and `debug.py` are the BEV side; the loop's hooks (`trainkit.loop.Hooks`) are how another project adds its own.
```
python -m trainsuite start training/configs/t20_base.json --id t20.diag --preset small --set split=small --set steps=600 --set grad_diag_every=100 --set profile_steps=4 --no-card
python -m trainsuite --root $R status t20_diag                        # the module table and the time split
```
`start` marks `profile_steps` as NEW (the base file does not list it); it is a `TrainConfig` field, so the check accepts it. The trainer accepts a changed `probe`, `profile_steps` or `grad_diag_every` on a resume, but `start` refuses a different config for an id that has a run: use a new id.

## 7. The training queue
```
python -m trainsuite queue                                            # (output shortened)
# GPU slots (2, ../bev_fusion_lane_center/experiment_execution/runs/.gpu_slots):
#   slot 0: t20.base_s0  (pid 1520792, since 2026-11-03T16:06:03)     slot 1: free
# queue (the first goes first): empty
# RAM: 12226 MB available, the gate asks for 10000 MB on 3 looks in a row: a run could start
# on hold (only the person's words clear a hold: hold.py clear ID --words ...):  profile.t10  speed.d  t16.ctl_s0 ...
```
**It exists** (since 2026-11-01). Every `run.py --gpu` is a launcher process that waits in one first-come-first-served line, takes one of the 2 GPU slots (the lane project counts against the same two) and starts only when the RAM gate lets it (`ram_mb` 10000 in `project.json`: two trainings run in parallel only when the box has RAM for both). A held id is refused. `--after <id>` waits for that run's done flag and is blocked when it failed or was stopped; a chain is `start ... --run-arg=--after --run-arg=<id>`:
```
python -m trainsuite start training/configs/t20_base.json --id t20.demo3 --preset small --set split=small --no-card --run-arg=--after --run-arg=t20.demo2
#   ... run.py --id t20.demo3 --gpu --split train --no-card --after t20.demo2 -- bash temp/train/run_ar.sh temp/compose/t20_demo3.json
```
**What it does not do:** there is no persistent list that starts runs by itself. The waiting launchers are processes of this session: a WSL shutdown or a reboot ends them, and the same commands start them again. The to-do list is `bev_fusion/records/schedule.csv` (`python3 tools/schedule.py show | next`, states read from the run records); whoever works the list starts the items with `start --go`. A tool that did start ready items (`tick.py`) was removed under D32 (it started 0 of 61 runs). If you want it, it is about 30 lines on top of `schedule.py next` and `start --go`.

## 8. Simple speed-up chances, ranked
The step is **host-bound**, not GPU-bound: in a 70-step profile of `t10_dual_s0` (accumulation 1, 2.66 s per step on a calm box, `speed.a3`) the GPU kernels were busy 33 % of the time, the main thread 85 %, and every step did 691 `.item()` calls, 1283 stream synchronisations and 1779 host-device copies (`speed.b`). The current chain (accumulation 2, 6 loader workers) takes 3.6 - 3.9 s per step: `t16.full_s0` median 3.62 s over the last 100 logged rows, 7 % of it waiting for data; `t20.small_s0` 3.86 s, 6 %. Measured so far, `bev_fusion/records/LEDGER.md` rows `speed.*`:

| rank | lever | measured | not measured |
|---|---|---|---|
| 1 | fewer host synchronisations (log every N steps, no `.item()` in the loop) | the counts above. Already cut, with the same numbers: one read-back per clip, multi-tensor EMA, targets from a window; the run with the read-back cut (`speed.b`) stayed inside the spread of the baseline (2.27 - 2.65 s against 2.44 - 3.14 s per step) | what the remaining ~690 syncs cost: `profile_steps` on a run of the current chain, then the top operators in `profile.txt` |
| 2 | larger micro-batch, fewer accumulation steps | t1 chain, effective batch 16, idle GPU: 4 x 4 = 2.5 s per step (4.4 GB), 8 x 2 = 2.1 s (8.6 GB), 2 x 8 = 5.3 s (`training/configs/README.md`) | the current chain: 3 x 2 peaks at 8.8 - 9.3 GB of the 11 GB allowed, so there is little room; BatchNorm sees other batches, so this changes the training, not only its speed |
| 3 | CUDA graphs / `torch.compile` of the trunk (43 of 78 GMACs of the cached chain; static shapes in training) | nothing: never tried | compile time, the dt loop, the pinned code snapshot |
| 4 | more loader workers | data wait is 4 % (`speed.a3`) to 7 % (`t16.full_s0`) of a step: at most that much to win | each worker costs RAM (one run takes about 10 GB) |
| tried | bf16 + `cudnn_benchmark` (`speed.c2`) | 2.71 against 2.66 s per step (no gain), peak memory -40 %, gradient norm median 865 against 34, loss +8 %: it changes the training | `tf32`, `pin_memory`, `prefetch_factor` and fp16 with a loss scale are `TrainConfig` switches, not measured alone |

Measure a change on a short run with a quiet GPU (the same config on a busy box ran at 7.7 s per step, `speed.a`, against 2.66 s on a calm one, `speed.a3`): `python -m trainsuite --root $R speed <base run> <changed run> --skip 10` puts them side by side. Three of the levers wait as held runs (`speed.d/e/f`).

More: [EXTENSIONS.md](EXTENSIONS.md) (the run directory contract, all checks, kill rules, early advice, `compose`, `prepare`, what is not built), the engine contract [../trainkit/README.md](../trainkit/README.md).
