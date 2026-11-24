# Provenance: what a result card and a result package record

Built; the collector is `tools/provenance.py` (`collect`, `compare`, `manifest`, `repro`), the result package and the clean-up of derived data are `tools/package.py`, the verified chain network -> run -> code -> data -> evaluations is `tools/lineage.py`. This page is the reference of the record; usage is in the docstring of each tool.

## The block `provenance` of a card
Written into `card["provenance"]` by the BEV evaluator (`bev_fusion/evaluation/bev_suite.py eval`); the lane evaluator and `Evaluation` write none yet (the lane cards carry no block, `decide.py` then treats the values as unknown). The schema text is in `schemas/card.schema.json`.
- `gt`: version, signature, res, ss, options (with `boxes_at`), and the hash of the store's `index.json`. The signature is the canonical one: the store, not the code signature (a generator version that makes the GT of an older one reads that older store).
- `evaluator`: commit, the list of files that differ from it (`dirty`; empty = clean), `code_hash` over the evaluator's source files, the metric settings. The hashes of the evaluator and of the checkpoint are taken at the start and at the end of the evaluation; a difference gives `dirty = "unknown"` and `changed_during_run`.
- `data`: nuScenes version, split name, number of frames, `frame_hash` of the sorted frame tokens, and `manifest_hash` of `data/NuScenes/manifest.json` (made by `data/NuScenes/tools/make_manifest.py`).
- `subject`: run id, checkpoint file hash and step, config hash, the GT version and signature it was trained on.
- `environment`: python, torch, CUDA, gpu, host; the seed.
A collector failure writes `{"schema": 1, "error": ...}` and a warning, never a card without a trace.

## The record of a training run
`provenance.py manifest <run id>` writes `<problem>/results/<id>/provenance.json` from the run record and the run folder: the GT it trained on, final checkpoint and step, config, seed, run status, code tag, and the model bundle (the files that `model.py` and `losses.py` import, by static import closure inside the run's own code snapshot, their hash and a copy in `<run folder>/bundle/`; what the bundle does not carry - repository code such as `GtGenerator`, files that do not parse, dynamic imports, a run that is not final - is listed, not hidden). `run.py` stores the code tag, commit, `code_hash`, `code_drift`, python and the command in `run.json`; the trainer stores the GT signature in `gt.json`.

## Rules
- `decide.py` refuses cards of one comparison that differ in `gt.version`, `gt.signature`, `evaluator.code_hash`, `data.frame_hash` or `data.manifest_hash` (`provenance.COMPARE_KEYS`), unless the card says why (`evaluate.allow ["provenance"]`).
- A card whose evaluator is not clean (or was not checked) is a blocker; `unknown` on either side of a key is "cannot tell", and `compare` exits 2, never 0.
- Old cards keep what they have; missing fields read "unknown", never a guess. A checkpoint over 64 MB is hashed by its first and last 4 MB plus its size (marked): a change in the middle is not seen.

## Reproduce a result
`provenance.py repro <run id>` prints what to check out and run: the code tag (`bf/<id>` or `lcr/<id>`), the GT tag of the GT version (`autogt/vN`) and the rebuild command if the store is gone, the split and frame hash, the evaluation command from `run.json`, and what differs on this machine (python, torch). `--repeat` checks the tag out in a worktree under `temp/repro/`. `lineage.py verify` checks every link of the kept networks.

## Git: what reproduces a run, and what keeps the history small
A run is reproduced from three places; only the first two are git, and only the first is pushed by default.

| place | holds | size rule |
|---|---|---|
| `main` (pushed) | code and text records (cards, ledgers, decisions, run records, metric cards, model configs) | a file at most 1 MB (the ledgers 3 MB), no data / binary / log type, no scratch: refused by `commit_paths.py` and `check_git.py` |
| run tags `lcr/<id>`, `bf/<id>` (one per run, local) | the commit `run.py` makes at the launch: the code folders as they were, plus the `.py` / `.sh` of the scratch folder (each at most 1 MB, at most 2000) | a tag stores only what differs from `main` (kB per run); nothing else that a `.gitignore` rule ignores goes in (every `temp/` and `_to_delete/` folder is ignored, also one inside a code folder, 2026-11-24: one json in `evaluation/temp/` had been swept into 443 run tags). Keep scratch scripts in the project's scratch folder: a `.py` in a `temp/` inside a code folder counts in the run's `code_hash` but is not in its tag |
| outside git (D: and the WSL home) | raw data, GT stores, caches, checkpoints | named by hashes (`data/NuScenes/manifest.json`, the GT signature, the checkpoint hash of the provenance) and made again by recipes (`data/rebuild.json`, `GtGenerator`) |

- Reproduce: `provenance.py repro <run id>` (above). The run tags are kept (current practice; deleting one is the owner's decision, hard rule 6); the local-only ones are also in a bundle on D: (`archive/lcr_local_tags_2026-11-24.bundle`, D143); they go to the remote only on the owner's word, and only the clean ones (`check_git.py` names the held-back tags and the file that holds each back).
- Prevent: the per-file and per-type rules above refuse the big single file at the commit and before a push. A file that is made again from a source (a table built from the ledger, a report's html) is not committed; the source is.
- Alarm: `check_git.py` prints a size line: the history of all refs on disk (budget 1 GB, GitHub's recommendation; its hard limit is 100 MB per file) and what `main` added in the last 30 days (budget 100 MB) with the biggest files and folders of that window. 2026-11-24: 50 MB in all, `main` +38 MB in its first month, spread over many text records (the biggest folders: the lane `experiment_execution/` 7.4 MB and `experiments/` 7.2 MB; the biggest single file is the run ledger, 3.2 MB in all versions, packed). When a budget is passed: find the file in the line, stop committing what is generated, merge records that should be one file. Never rewrite history, delete tags or force-push to get under a budget: that is the owner's decision (hard rule 6).
