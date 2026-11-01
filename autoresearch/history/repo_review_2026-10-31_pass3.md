# Repository review 2026-10-31, pass 3: bugs, architecture, working guide

Date: 2026-10-31. Base commit of the code reviewed: `37cefea` (plus the uncommitted work of other chats, named where it matters).
Earlier reviews of the same day: `repo_review_2026-10-31.md` (pass 1, section 8 = what was done) and `repo_review_2026-10-31_pass2.md`.
This pass was written by a third reviewer; this file is its text after a check against the code and the repair of the bugs that mattered
(section 2). Finding ids are `P3-B1` ... `P3-B9` (pass 2 has its own B ids).

## 1. Summary

| id | where | what | active or latent | status |
|---|---|---|---|---|
| P3-B1 | `bev_fusion_lane_center/training/trainer.py` `fit` | a finite loss with a non-finite gradient was applied (bf16 / fp32 path: weights became NaN); a run in which every step was skipped finished `done` | latent: only `trainer_selftest.py` calls `fit` today (`training/README.md` says new work should use it) | **fixed**, test |
| P3-B2 | same | `itertools.cycle(loader)` replays the batches of the first pass | latent (same reason) | **fixed**, test |
| P3-B3 | `bev_fusion/autogt/cli.py` `build` | the cache folder is named by the generator signature (no resolution, no supersampling); the index that guards a resume was written only at the end, so an interrupted build was resumed at another resolution | active when a build is interrupted and restarted with another `--res` | **fixed** (provisional index, shape check), test |
| P3-B4 | `bev_fusion/training/train.py` `trainable_state` | camera weights are never in the checkpoint `state`, also when `freeze_camera` is false | not yet (`freeze_camera` is true everywhere), **active as soon as the trainable image features of D26 run** | **fixed**, test |
| P3-B5 | `bev_fusion/training/nuscenes_dataset.py` `gt_signature` | the GT cache key holds resolution, supersampling and generator signature, not the grid extents | active with any non-default grid | **fixed** compatibly (default names unchanged), test |
| P3-B6 | `autoresearch/trainsuite/runs.py` `read_log` | after a training rollback the validation rows of the superseded suffix stay; a kill rule can fire on them (`watch --enforce` stops a healthy run) | active | fixed by another chat, uncommitted when this was written |
| P3-B7 | `autoresearch/tools/ledger.py` `missing` | an older terminal ledger row hides a newer run record from `sync` | rare (known as L1 of `ROADMAP.md`, the ledger one) | another chat edits `ledger.py`, uncommitted |
| P3-B8 | `bev_fusion/autogt/video.py` `run` | a failure leaves the temporary directory with the pictures; `cv2.imwrite` failures are not noticed | low | **fixed**, test |
| P3-B9 | `datasets/NuScenes/tools/verify_s3_etag.py` | the multipart part size is guessed as the smallest whole MiB that gives the part count | low (a one-off script, says so itself) | open |

Not a finding any more: the opening paragraph of `bev_fusion/README.md` was reported as stale; it was brought up to D23 in `a7820ec`.

## 2. The bugs, what is wrong, what was done

All nine were confirmed by reading the code at the place named. The reproductions of the first reviewer (synthetic inputs) were not
re-run; instead each repaired bug has a regression test that fails on the code of `37cefea` and passes now.

**P3-B1 and P3-B2, the lane trainer.** `fit` checked only `torch.isfinite(loss)`. A finite loss can have a non-finite gradient (a square
root at 0): with bf16 or fp32 (`GradScaler` disabled) `scaler.step` applied it. The new self-test case shows the effect on the old code:
one NaN gradient made every later loss NaN, 30 steps were skipped, and the run still wrote status `done`. The fp16 `GradScaler` already
skips such steps itself, so the fp16 path is unchanged. The bev_fusion trainer (`train.py`, gradient norm check and `max_bad_steps`, the loader
iterator renewed on exhaustion) already had both guards; the lane trainer now has the same:
- the gradient norm is measured always (`clip_grad_norm_` with `inf` when clipping is off); a non-finite norm skips the step like a non-finite loss;
- with the fp16 scaler, a step the scaler skips itself (its scale falls) is counted the same way and is not an update;
- `max_bad` (default 20) non-finite steps in a row, or a run that never updated a weight, raise `RuntimeError`: status `failed`, no `.done`, no `model.pt`;
- a skipped step still goes through the checkpoint and evaluation cadence (before, a skipped step on a multiple of `ckpt_every` lost that checkpoint);
- the status of a finished run holds `updates` (optimizer steps made) next to `skipped`;
- `_endless(loader)` iterates the loader anew every pass (and refuses an empty loader, saying that a one-shot iterator cannot be passed).
Skip-and-count itself is documented design and stays. Open: `max_bad` bounds only consecutive skips, a run with most steps skipped but never `max_bad`
in a row still ends `done` (the count is in `status.json`). Tests: `bev_fusion_lane_center/training/trainer_selftest.py`, cases 5-7 (a stand-in scaler on the CPU for fp16;
bursts of `max_bad - 1` finish, a failing run stops at step `max_bad`).

**P3-B3, an interrupted autogt build.** `build` now writes `index.json` with `"complete": false` before the first keyframe (atomically) and
with `"complete": true` at the end; `check_resume` refuses another `res` / `ss` against it. A folder of an interrupted build from before this
change has no index: one stored keyframe's `drivable_area` shape is compared with the requested cell grid. The directory name is NOT
changed: signature `15efbde2d1` is final and the trainval store is built under it (a changed key would orphan 850 sequences, about 5.5 h).
`build` writes the provisional index on every start, also on a resume of a completed folder; `report` calls `check_resume` too (it writes the
same folders). `clean --stale-caches` and `--all-caches` leave a folder with a provisional index alone for an hour after its last write (this also
closes the data-loss case of pass 2, C9, where `clean` could move a folder still being written). Tests: `bev_fusion/tests/test_gt_cache_targets.py` F5
(the helpers, `build` itself through a fake generator, and `clean.collect`).

**P3-B4, camera weights.** `trainable_state` dropped every `camera.*` entry. The EMA shadow holds them anyway (it takes every floating
point entry of the state dict), so evaluation with EMA weights was right; what broke was the resume of the live weights: training went on
from the pre-trained file while the optimizer state of the trained camera was restored. Now the entries stay in `state` when
`freeze_camera` is false, and `load_checkpoint` refuses a checkpoint without them for such a model (a clear message instead of a silent
restart). Frozen runs (all runs so far) are unchanged byte for byte. Test: F7.

**P3-B5, GT cache key.** `gt_signature()` (the suffix of the cache files, also recorded in `gt.json` and the evaluation manifests) is
`<res>_<ss>_<generator signature>`. It now carries `_x<x_min>_<x_max>_y<y_min>_<y_max>` after the resolution when the grid is not the default
front grid (x 0..60 m, y -26..26 m); the default keeps all existing names. The first draft proposed hashing all grid parameters: that
would have renamed every cache file. `evaluation/bev_adapter.py` had its own copy of the formula; it now calls the dataset's method (one identity
for the cache, `gt.json` and the evaluation manifests). The independent code review found that the datasets are always built with the default grid
while `cfg.model["grid_kwargs"]` changes the model's grid only: a model on x 10..70 would have trained on labels cut for x 0..60, same shapes, no error.
`train.check_grids` now refuses that. No config uses `grid_kwargs` today. Test: F4.

**P3-B6, superseded evaluations.** `read_log` deduplicated the training stream and the evaluation stream separately, so after a rollback
to step 500 the step-1000 evaluation of the discarded suffix stayed and could trigger a kill rule (`watch --enforce` stops the run). The fix
pops the evaluations from the restart step on when a training row restarts the stream, with a test (`trainsuite/selftest.py`). It is the
work of another chat and was uncommitted when this was written.

**P3-B7, the ledger.** `missing()` repairs only planned / running / stalled rows, so an old `failed` row of an id that was run again and
finished is not replaced by `sync`. It is the ledger L1 of `ROADMAP.md` (rare, accepted). The id clashes with the L1 of pass 1
(`graph_from_vertices`): write "ROADMAP L1 (ledger)". Another chat has an uncommitted change in `ledger.py`.

**P3-B8, the video temporary directory.** `autogt.video.run` removes its temporary directory in a `finally`, creates the output folder,
and raises on a failed `cv2.imwrite`. Test: F6.

**P3-B9, the ETag verifier.** It is a one-off (its docstring names a stale path, `temp/loop19/`). File size and part count do not give
the part size. A fix that helps: try the usual part sizes (5, 8, 16, 64 MiB and the smallest compatible one) and accept any that matches, or a
`--part-size` argument. Not done.

## 3. How this was checked

- Lane `trainer_selftest.py` (the new cases 5-7 included): passes; against the trainer of `37cefea` the new case 5 fails as described above.
- `bev_fusion/tests/test_gt_cache_targets.py` F1 - F7 (CPU, no data): all pass; F4 - F7 fail on the code of `37cefea`.
- An independent `ar-reviewer` pass (mode code) over the diff found no crash, shape or gradient bug, and these gaps, all closed above except the first of the open ones:
  the fp16 scaler's skips were not counted (and the docs said they were), "in a row" was untested (mutants that deleted the reset passed), the F5 test covered
  helpers and not `build` / `clean.collect`, `report` and `--all-caches` bypassed the new guards, the model grid could differ from the dataset grid, and the
  evaluation kept its own copy of the identity formula. Open: a bound on the skipped fraction of a run.
- `python -m autogt selftest` passed with the provisional index (signature still `15efbde2d1`); the `training/train.py` selftest passed on the code before `check_grids` was added (the
  rest of that file is unchanged and `check_grids` has F4); `bev_selftest.py` passed before the last edits and was not re-run on the final code (RAM: the first `train.py` selftest used
  7 GB and an out-of-memory kill took `t10.dual_s1` at 00:09, so no further heavy check ran that night). The final code of the fixes passed F1 - F7 and the lane `trainer_selftest.py`.
- Where the fixes were committed: the autogt part (`build` / `check_resume` / `write_index`, `clean._running`, `report`, `video`) went into the autogt chat's commit `4672b3f`, the `train.py` part
  (`camera_frozen`, `check_grids`) into `32ab9be`; the rest (lane trainer, GT cache key, evaluation identity, tests, plan and README edits, this file) is the commit that adds this file.
- The first reviewer's own checks (not re-run here): Python / shell / JSON / PowerShell syntax of the whole tree; research-tool selftests; lane builder reference tests 18/18;
  graph simplification 9/9; lane TEST guard 4; BEV GT contract tests; visualization orientation tests; `bev_selftest.py` (synthetic world) 9/9.
- The inventory of the first reviewer (4,292 paths; 860 Python files parsed, 445 shell scripts, 1,510 JSON, 4 PowerShell) was taken on an unpinned working
  tree. Re-counted with `git ls-files -co --exclude-standard`: 4,298 paths, 955 `.py`, 493 `.sh`, 1,512 `.json`, 4 `.ps1`. The difference in
  `.py` and `.sh` is not explained (other scope, or files that do not exist); the next review should name its command and commit.
- Not done: real training, TEST evaluation, downloads, installations, GPU benchmarks. Two files that did not parse under Python 3.10 parse under the product's Python 3.12.
- Dataset and checkpoint binaries were not inspected. This is a repository-wide inventory plus a deep look at the critical paths, not a
  line-by-line review of every historical experiment or literature note.

## 4. Earlier findings that concurrent fixes closed

Another chat changed research tooling and evaluation storage while the review ran (commit `e203d06`, "enforce what a KEEP rests on"). Rechecked and
closed there: the cross-split and denominator mismatch of two cards, the pairing of seeds by identity (reordered seeds give `INCONCLUSIVE`), infinite
metric values, changed explicit frame selections on an evaluation resume, and an incomplete trailing record of an evaluation store.

Provenance, what is still open: a recorded snapshot does not prove that the snapshot ran. The launcher records code drift; the bev_fusion wrapper
`temp/train/run_ar.sh` trains from `git archive` of `training/` at `$AR_COMMIT` but imports AutoGT from the live tree. Since D24 the GT signature is
written to `runs/<name>/gt.json` and `bev_suite.py compare` refuses runs with different signatures: that detects a mix, it does not prevent it.
Executing the training code and the GT generator from recorded versions together is open.

## 5. Architecture

Two perception products, one research controller, shared raw data. Keep the four-way split; it needs no larger framework first.

```text
                 Research controller (autoresearch/)
         METHOD.md phases P1 research -> P2 hypotheses + plan -> P3 loop -> P4 close
                              |
                  +-----------+-----------+
                  |                       |
        Lane-graph product        BEV product
        (bev_fusion_lane_center)  (bev_fusion)
                  ^                       ^
                  +-----------+-----------+
                              |
                    Shared nuScenes data (datasets/NuScenes)
```

**Research controller** (`autoresearch/`, a control plane, not a network):
- `METHOD.md`, `agents/`, `templates/`, `prompts/`: phases, roles, cards, loop prompt.
- `tools/`: launcher `run.py` (snapshot commit and tag per run, GPU slots, retries), ledger, schedule, `decide.py`, card / experiment checks, guard hook.
- `evalsuite/`: measurement for any network. Source (scenes, frames, inputs, ground truth), Subject (runs the model), Task (metrics, checks, views), RunStore
  (per-frame counts, so any set of frames reduces to numbers without the network). `evalsuite/comparison/` is the one reviewed place where a
  prediction is compared with the ground truth (maps, boxes), with a mutation check. Problem-specific tasks live with their problem
  (the lane task: `bev_fusion_lane_center/evaluation/lane_task/` and `lane_suite.py`; the BEV task: `bev_fusion/evaluation/bev_task.py`).
- `trainsuite/`: reads training logs, health, kill rules, cooperative stop. The optimizer loops belong to the products.

**bev_fusion** predicts maps, objects and motion from camera and radar. Camera encoder -> depth-weighted geometric lift (calibration, no learned virtual
camera) -> camera temporal fusion; radar sweeps aligned in time and pose -> radar representation; fusion -> BEV trunk -> time mover -> semantic / object /
motion heads decoded at requested time offsets. Default front grid 60 m x 52 m at 0.5 m = (120, 104) cells. Since D23 it runs under `run.py` with its own
`bev_fusion/project.json` (tag prefix `bf`, records in `bev_fusion/records/`). Entry points: `training/model.py` (assembly), `training/train.py` (loop),
`training/nuscenes_dataset.py` (sensor data and target cache), `evaluation/bev_adapter.py`. AutoGT (`bev_fusion/autogt/`) is a separate supervisory-data
dependency, not a network part: maps, annotations, ego motion and radar in, ground truth out; its generator signature is part of every GT cache key.

**bev_fusion_lane_center** produces a sparse connected lane graph attached to an SD-map route: cached camera + radar BEV features -> dense lane evidence ->
sparse graph builder (seeds, rays, survival, children / junctions) -> assembly and pruning -> temporal / map stages -> route head along graph edges.
The learned builder is differentiable; the CPU graph assembly and output stages are a separate boundary, so the deployed pipeline must not be called
fully differentiable without that qualification (rule steps in the delivered graph are a named gap).

**Shared data**: `datasets/NuScenes/` holds recipes and checks; raw data and checkpoints are outside Git, derived data belongs to the product that
makes it. The WSL disk image is infrastructure, not a folder to reorganise.

## 6. Working guide

Start in the right place:

| task | start with |
|---|---|
| research orchestration | `autoresearch/README.md`, `METHOD.md` |
| lane-graph work | `bev_fusion_lane_center/HANDOVER.md`, `PROBLEM.md`, `CLAUDE.md` |
| BEV network | `bev_fusion/README.md`, `plan.md`, `training/README.md`, the config |
| ground truth | `bev_fusion/autogt/AGENTS.md` |
| evaluation, debugging | `bev_fusion/evaluation/README.md`, `bev_fusion_lane_center/evaluation/README.md`, `autoresearch/evalsuite/README.md` |

Rules that save trouble (the commands themselves live in the documents named, not here, so there is one copy to keep current):
- Read the state before editing: HANDOVER, DECISIONS, PLAN, `schedule.py show`, `ledger.py tail`. Use `tick.py --dry-run` first; without it ready work starts.
- Launch every training through `run.py`. For bev_fusion: `AR_PROJECT=../bev_fusion`, `temp/train/run_ar.sh` (not `run_cfg.sh`, which always exits 0), in the
  background (`nohup`); the exact line is in `bev_fusion/records/README.md`. The launcher run id and the training directory name differ: monitor with the
  directory name (`trainsuite/README.md` has the commands; check whether a command takes a name or name:control pairs).
- The lane harness (`trainer.py`) and the BEV trainer (`train.py`) are different implementations; logs, checkpoints and configs are not interchangeable.
- Evaluate through the product's standard entry point (lane: `evaluate.py <registry name> --compare <control>`; BEV: `bev_suite.py`), pictures and
  sanity controls first. A healthy loss does not make a good perception product.
- Shared machine: look at running jobs and code ownership first; no edits of shared modules or wrappers while a job uses them; no full GPU suites during another
  chat's training or timing window; TEST needs a recorded permission; one-off analysis stays in `temp/`.

## 7. Order of work, by id

1. Success semantics of training: P3-B1 (done).
2. Identity of data and caches: P3-B3, P3-B5 (done, compatible); open: GT from a recorded version together with the training code.
3. Recovery: P3-B4 (done), P3-B6 and P3-B7 (other chat, commit pending).
4. Immutable dependencies (see section 4).
5. Public entry points: fewer flat-import search paths and implicit snapshot imports.
6. Living documents: keep history, mark the authoritative workflow. Decide whether the lane `trainer.py` gets its first user (the README says retrofit `p17_mapft.py`) or goes.
7. P3-B9 if the script is used again.

## Conclusion

The highest-value work is the contract around which data were used, which code ran, what a finished run means and how a disrupted run resumes. The
fixes of this pass cost little because the bev_fusion trainer already showed the pattern.
