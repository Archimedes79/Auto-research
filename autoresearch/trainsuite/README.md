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

## 0. The interface (start here)
One class, `Training(project)`, with plain dicts in and out; every method is also a command (the pattern of all suites: [../INTERFACE.md](../INTERFACE.md); the code: `api.py`, `apicli.py`).
`from trainsuite import Training` in Python, `python -m trainsuite Training.queue_add job.json` on the command line (short forms: `queue_add`, `queue-add`; `Training.` is always needed for `status`, `stop`, `plot`, `compare`, which are older commands;
`python -m trainsuite methods` lists the methods with their arguments; `--json` prints exactly what the method returns).
A **job** is a json file (or the same dict) that says what to run:
```json
{"id": "t23.C_s1", "config": "training/configs/t23_lane_bce.json", "preset": ["fast"], "set": {"seed": 1}, "card": {"hyp": "...", "pred": "...", "rule": "..."},
 "after": [], "priority": 5, "kill": ["group.lines@1000 > 1.48"], "note": ""}
```
`no_card: true` instead of `card` is a diagnostic run (it never feeds a verdict). An unknown key is an error (a typo shows), keys starting with `_` are comments, `true` / `false` must be booleans.
`kill` rules are checked when the job is read and appended to the run's `kill.txt` once the launcher has survived its first seconds (where `watch --enforce` and `status` read them). `run`
honours `after` (run.py waits for those runs) and ignores `priority` and `note`, which belong to the queue. `run_args` are arguments for `run.py` itself, `no_check` skips the project's static
check, `prepare` builds a missing data store first, `project` names another project folder (default `AR_PROJECT`); a key of a runner job (`n`, `out`, ...) is an error in a training.
`i` below is an id, a run folder name or the number shown by `queue-list` (numbers shift when another chat adds an item: a script uses the id).

| method | command line | what it does |
|---|---|---|
| `check(job)` | `check JOB` | compose the config, the static check, the data check; starts nothing |
| `run(job)` | `run JOB` | start now, outside the queue; the same call again resumes |
| `queue_add(job)` | `queue-add JOB` | check, put it in the queue; the runner starts it when its turn comes |
| `queue_list()` | `queue-list` | the items in order, the running one, slots, machine lock, RAM, GPU memory, holds |
| `queue_get(i)` / `queue_remove(i)` | `queue-get I` / `queue-remove I` | one item in full / take out one that has not started |
| `queue_start()` | `queue-start` | make sure the runner runs |
| `queue_hold(words)` / `queue_resume(words)` | `queue-hold --all --words ...` / `queue-resume --all --words ...` | all trainings stop with a checkpoint / continue from it (`i` for one); the person's words are required |
| `status(i=None)` | `Training.status [I]` | state, ETA, loss, gradient, skipped steps, data, health, kill rules, speed; no `i` = the running one |
| `stop(i)` / `plot(i)` / `compare(a, b)` | `Training.stop RUN` / `Training.plot RUN` / `Training.compare A B` | checkpoint and stop one run / loss curves as png / speed side by side |

Mirko's sketch in these words: `Training.run(network.json)` = `run`, `queueAdd` = `queue_add`, `queueGetListOverview` = `queue_list`, `queueGet(i)` = `queue_get`, `queueRemove(i)` = `queue_remove`,
`queueStart` = `queue_start`, `queueHold` / `queueResume` = `queue_hold` / `queue_resume`, `queueGetStatus(i)`, `queueRunningTrainingId` and `queueRunningTrainingStatus` = `status(i)` (no `i`:
the running one; `queue_list()['running']` is its id), `QueueLogSpeed` = `compare` and the speed part of `status`.

The older words (`start`, `queue add ... --id`, `queue serve`, `queue drop`, `curves`, `speed`, and the old `status` / `stop` tables) work as before. The runner of a trained network has the same
interface in `autoresearch/runsuite` (`Runner`: `check`, `run`, `status`, `queue_add`; one queue for both). `python -m trainsuite.selftest_api` tests it (it also runs inside `selftest`).

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
- **The whole line in one command** (section 7): `queue hold --all --words "<their words>"` stops what runs (a checkpoint each), keeps what waits from starting and pauses the runner. `queue resume --all --words "<their words>"` clears those holds and every stopped run **continues from its checkpoint on the code of its first attempt** (`run.py --pin`: a changed working tree does not leak into a resumed run), in its old place of the line. `--fresh-code` is the hot-fix loop above. Both commands refuse without the person's words.

## 5. How to supervise
| command | answers |
|---|---|
| `status [RUN]` | where is every run (state, step, ETA, loss, the gradient norm now and against before, how many non-finite steps were skipped); for one run also the data it trained on, the gradient norm (first rows, now, peak and its step), the steps that were skipped, the loss groups, the module table, the time split, the last eval |
| `health RUN [--vs REF] [--kill RULE]` | is it sound: crash, stall, NaN / inf, skipped steps, spike, rising loss, exploding gradient, dead module, data-bound; `--vs` warns when the reference trained on other data; exit 1 on an alert or a kill hit |
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
#   slot 0: t20.base_s0  (pid 1520792, since 2026-11-03T16:06:03, loading: 40 of 180 s settle time, the next launcher waits)     slot 1: free
# machine lock: free                         (or: timing until 2026-11-04T09:00 (EXCLUSIVE: no other owner starts a run), phase timing)
# queue (the first goes first): empty        (a launcher the lock holds back is marked 'held back by the machine lock': it does not hold up the others)
# RAM: 12226 MB available, the gate asks for 10000 MB on 3 looks in a row: a run could start
# GPU memory: 8718 MB in use; a training starts only below 3000 MB: a training waits
# on hold (only the person's words clear a hold: hold.py clear ID --words ...):  profile.t10  speed.d  t16.ctl_s0 ...
```
**It exists** (since 2026-11-01). The view shows the **machine lock** (`machine_lock.py`: a measurement that needs the box alone; an exclusive lock holds every other launcher back, and its ticket in the line does not hold up the lock owner's own launches), and a slot that is still **loading** (see below). Every `run.py --gpu` is a launcher process that waits in one first-come-first-served line, takes one of the 2 GPU slots (the lane project counts against the same two) and starts only when the RAM gate lets it (`ram_mb` 10000 in `project.json`: two jobs run in parallel only when the box has RAM for both; with the memory gate below, two big trainings never overlap). A held id is refused. **Settle time** (`gpu_settle_s` in `project.json`, `--settle-s`, 0 = off, `AR_SETTLE_S` overrides; bev_fusion sets 180): a launcher waits while another one took its slot less than that many seconds ago, because that run has not loaded its data or allocated its GPU memory yet, so the RAM and memory gates would read the box too early (on 2026-11-04 two trainings started in the same second on the 12 GB card and slowed each other 13 times). It delays only launchers of a project that sets it, evaluations of that project included (a run that ends within the time frees its slot and delays nobody). **Memory gate:** the `trainsuite` block of `project.json` may set `"gpu_max_mb": 3000`; `start`, `queue add` and the queue runner then pass `--gpu-max-mb 3000` to `run.py`, so a training starts only when the card has less than that in use, i.e. when no other big job holds it (a hand-typed `run.py` line without the flag is not gated; the view shows the memory in use against the gate, and a waiting launcher says in its heartbeat what it waits for).
```
python -m trainsuite start training/configs/t20_base.json --id t20.demo3 --preset small --set split=small --no-card --run-arg=--after --run-arg=t20.demo2
#   ... run.py --id t20.demo3 --gpu --split train --no-card --after t20.demo2 -- bash temp/train/run_ar.sh temp/compose/t20_demo3.json
```
**The queue that outlives its launchers** (since 2026-11-03). A launcher waits as a process: a WSL shutdown, a kill or `--wait-after` running out ended every chain behind it (the chain behind `t17.data` went "blocked" on 2026-11-02). The queue is a persistent list plus one small runner:
```
python -m trainsuite queue add training/configs/t20_base.json --id t20.b --hyp "..." --pred "..." --rule "..." [--after t20.a] [--priority 3]
python -m trainsuite queue                                       # the view above plus the items: state, step, gradient norm now (x against before), skipped non-finite steps, health flags, --after, pin
python -m trainsuite queue hold --all --words "..."              # stop what runs (checkpoint), nothing new starts      (or: hold t20.b --words "...")
python -m trainsuite queue resume --all --words "..."            # everything continues from its checkpoint            (or: resume t20.b ...; --fresh-code)
python -m trainsuite queue drop t20.b                            # take an item that has not started out of the line
```
- `add` composes like `start` (same checks; a missing GT store is refused, build it first) and stores the **composed config with the item**: what is queued is what runs. The list is `queue.<project>.json` in the GPU slot folder (not in git; one file per problem, the slot folder is shared by the problems that share the GPU); a run's state is never stored, it is read from the run record and folder, so the list cannot go stale. An unreadable file is moved aside (`.bad.<time>`) and refused, never written over.
- The **runner** (`queue serve`, started by `add` and `resume`) looks every 30 s: it starts the first item whose `--after` runs are done and while no other item waits for a slot (the line keeps its order; `run.py` still owns the slots, the RAM gate and the holds). A failed or held `--after` run blocks its item (shown as `blocked by ...`). It ends after 10 minutes with nothing left to do; a WSL shutdown ends it too: `queue serve`, `add` or `resume` bring it back, everything else is in the files. It is not started at Windows boot (a scheduled task would be a system change: not installed).
- **Interrupted runs** (the launcher is gone and the heartbeat is 5 minutes old, `AR_STALE_S`): the runner restarts one **only when its launcher died in an earlier boot of the machine** (a WSL shutdown, a reboot), from the last checkpoint and on the code of its first attempt, at most 3 times in a row without a new logged step. A launcher somebody killed by hand in this boot is *not* restarted: the person stopped it, `queue` says so and `resume <id> --words ...` continues it. A trainer that outlived its killed launcher (an *orphan*, found by its config path in `/proc`) is left alone, never started twice. A run that **failed** (traceback, kill rule) is not restarted by itself either: look at it, then `resume`.
- `hold` and `resume` also work on a run the queue did not start: `hold` reads the live launcher's command line before the stop and keeps it; a run whose launcher is gone is rebuilt from its run record (`command`, card; `--ram-mb` and other launcher flags are the defaults). `hold --all` stops **trainings** only (a run folder of its own, or a command that names it): an evaluation, a data build or a timing job is reported as "not touched". `resume --all` clears only the holds of runs in the queue list, never an older hold such as `t10.dual_long_s0` (name it: `resume t10.dual_long_s0 --words ...`).
- The to-do list stays `bev_fusion/records/schedule.csv` (`tools/schedule.py show | next`); an item of it becomes a queue item with `queue add`. The old `tick.py` (removed under D32, 0 of 61 runs started) was a pass over that list that nobody ran; the runner is one process that keeps looking. `python -m trainsuite selftest` runs the queue against a toy trainer in about a minute (`TS_SELFTEST_NO_QUEUE=1` leaves it out).

## 8. Simple speed-up chances, ranked
The step is **host-bound**, not GPU-bound: in a 70-step profile of `t10_dual_s0` (accumulation 1, 2.66 s per step on a calm box, `speed.a3`) the GPU kernels were busy 33 % of the time, the main thread 85 %, and every step did 691 `.item()` calls, 1283 stream synchronisations and 1779 host-device copies (`speed.b`). The current chain (accumulation 2, 6 loader workers) takes 3.6 - 3.9 s per step: `t16.full_s0` median 3.62 s over the last 100 logged rows, 7 % of it waiting for data; `t20.small_s0` 3.86 s, 6 %. Measured so far, `bev_fusion/records/LEDGER.md` rows `speed.*`:

Where the launches come from (counted on CPU for the t20 chain, 2026-11-03, `bev_fusion/records/speed_options_2026-11-03.md`): a step is about 43-50 thousand operators (about 85 microseconds each at 4.09 s per step), the trunk, the part the GPU is busy with, is 5 % of them. 22 % are the 9 splats, 16 % the map loss terms, 11 % the two heads, 8 % the object-head loss, 7 % the motion block; the `for dt` loop runs everything 3 times and both heads call the same mover, so 3 of the 6 splats per keyframe repeat work.

| rank | lever | evidence | not measured |
|---|---|---|---|
| 1 | fewer launches: one mover call per dt for both heads (-9 % ops), one `index_add` instead of the 4-corner loop in `splat` (-13 %), the ~35 sync-like ops per keyframe that can go bit for bit (`motion_term` masks, `radar_encoder`, `lift_pd` `.tolist()`, `w_obj` copies), one host-to-device copy of the GT per key instead of per sample (-1,300 copies per step) | operators counted per section (CPU), sync sites from a dispatch trace; module code is open under FROZEN.md, the `flip` / `isfinite` syncs are frozen | the seconds: `profile_steps` + `speed` on a quiet GPU. The old cut (one read-back per clip, multi-tensor EMA) stayed inside the spread (`speed.b`). **Built behind default-off flags (D53, preset `fast`) and measured on the t20 chain: all four together 3.51 s per step against 3.86 / 3.69 s for two plain runs (-5 to -9 %, -1.2 GB peak); one lever alone is inside the 4 % noise** (`records/speed_options_2026-11-03.md` section 7) |
| 2 | `torch.compile` of the parts that trace without a break: trunk + fusion, `map_terms`, the motion-block net, `splat_bev` | `dynamo.explain`: forward 34 breaks, loss 39 (per-box `float()`, `.tolist()`, dynamic shapes); the listed parts have none; trunk + fusion is only 5 % of the operators | compile time, the pinned code snapshot, whether the GPU time falls |
| 3 | the 3 dts as one batch of 3B (-40 % ops) | counted | **changes the training**: BatchNorm statistics over dt (open finding #5), so a new baseline and Mirko's word |
| 4 | larger micro-batch, fewer accumulation steps | t1 chain, effective batch 16, idle GPU: 4 x 4 = 2.5 s per step (4.4 GB), 8 x 2 = 2.1 s (8.6 GB), 2 x 8 = 5.3 s (`training/configs/README.md`) | the current chain peaks at 8.8 - 9.3 GB of 11 GB; BatchNorm sees other batches |
| 5 | more loader workers / targets in the workers | data wait 4 % (`speed.a3`) to 7 % (`t16.full_s0`): at most that much; `w_obj` depends on the splat mass, so the targets cannot leave the main process | RAM per worker (one run takes about 10 GB) |
| no gain | fused AdamW (< 1 % of the launches), `cudnn_benchmark`, TF32 (risky for coordinates) while the step is host-bound; bf16 + `cudnn_benchmark` (`speed.c2`): 2.71 against 2.66 s, peak memory -40 %, gradient norm median 865 against 34, loss +8 % | | fp16 with a loss scale is built (`speed.e` waits as a held run) |

Measure a change on a short run with a quiet GPU (the same config on a busy box ran at 7.7 s per step, `speed.a`, against 2.66 s on a calm one, `speed.a3`): `python -m trainsuite --root $R speed <base run> <changed run> --skip 10` puts them side by side. Three of the levers wait as held runs (`speed.d/e/f`). To compare two settings of the SAME model (the switches of D53 / D54, `pin_memory`, workers), the in-process A/B timer `bev_fusion/training/ab_timing.py` needs about 7 minutes for two arms (4 rounds) and sees less drift than two runs (blocks of steps in the order A B / B A in one process, no checkpoints); a different model (a built-in option) still goes through a run and `speed`.

More: [EXTENSIONS.md](EXTENSIONS.md) (the run directory contract, all checks, kill rules, early advice, `compose`, `prepare`, what is not built), the engine contract [../trainkit/README.md](../trainkit/README.md).
