# Reproducibility: what a result package records (design, 2026-11-01)

Mirko, in chat: more reproducibility; when a result package of an experiment is made it should say which autoGT was used in which mode and which evaluation code version, so that the experiment can be reproduced later. This page says what is recorded today, what is missing and where it goes. Status: the collector, `compare` and `repro` are built (`tools/provenance.py`, with a selftest, 2026-11-01); the evaluation card writer calls it (`bev_fusion/evaluation/bev_suite.py eval`, commit 8573172: `card["provenance"]`; evaluator and checkpoint hashes are taken at the start and at the end of the evaluation, a difference gives `evaluator.dirty = "unknown"` and `changed_during_run`; a collector failure writes `{"schema": 1, "error": ...}` and a warning); the `provenance` field in the card contract and the refusal in `decide.py` are written by the Training chat (commit pending); the entry in `run.json` is left to the trainer and evaluator (the launcher's python is not the trainer's python). `compare()` treats an error-only block like `decide.provenance_check` (a blocker). Training side (Mirko: only the network and the losses form a bundle): `provenance.py manifest <run id>` writes `<run records>/<id>/provenance.json` from the run record and the run folder (GT it trained on, final checkpoint and step, config, seed, run status, code tag) and the model bundle: the files that `model.py` and `losses.py` import (static import closure inside the run's own code snapshot `<run folder>/code`; the training loop and the data loaders are not followed but listed when the network imports them), their hash and a copy in `<run folder>/bundle/`; what the bundle does not carry (repository code such as `autogt`, files that do not parse, dynamic imports, a run that is not final) is listed, not hidden. The launcher does not call it yet (run.py end of run: Training chat); `repro <run id>` reads the manifest.

## Today (checked on `t10.dual_s1.eval` and the training run `t10_dual_s1`)
| where | what it holds |
|---|---|
| `run.json` of a run | `commit`, `tag`, `code_hash`, `code_drift` (files that differ between the snapshot and the working tree), python, gpu, host, command |
| `card.json` of an evaluation | `commit` (`+dirty` when tracked files differed), `protocol: {gt_signature}` |
| `gt.json` of a training run | `gt_signature` (e.g. `0.5_4_15efbde2d1`) |
| GT store `data/autogt/<signature>/index.json` | `signature`, `gt_version` (v5, since the D29 versioning), `version` (v1.0-trainval), `res`, `ss`, `options` |

## Gaps
1. The readable `gt_version` and the generator mode (`options`, for example `boxes_at`) are in the GT store, not in the card or in the training record.
2. `card.commit` is the repository commit, not a hash of the evaluator's own code. `t10.dual_s1.eval` had `code_drift` n = 4 (evalsuite and bev_selftest files differed from the commit): the commit does not pin what ran.
3. No hash of the frame list: which tokens were scored is not in the card.
4. The card does not name the model: run id, checkpoint hash and step, config hash, the GT version it was trained on.
5. Software: python only; no torch / CUDA version.

## The record: one block `provenance`, written into `card.json` and `run.json`
- `gt`: version, signature, res, ss, options (with `boxes_at`), and the hash of the store's `index.json`.
- `evaluator`: commit, the list of files that differ from it (empty = clean), a hash over the evaluator's source files (`evaluation/` and `autoresearch/evalsuite/` of the problem), the metric settings.
- `data`: nuScenes version, split name, number of frames and the hash of the sorted frame tokens; the hash of `data/NuScenes/manifest.json` (archives, tables, sensor-folder fingerprints; made by `data/NuScenes/tools/make_manifest.py`) with the scene and sample counts, compared as `data.manifest_hash`.
- `subject`: run id, checkpoint file hash and step, config hash, training GT version and signature.
- `environment`: python, torch, CUDA, gpu, host; the seed.

## Rules
- `decide.py` and the comparison tools refuse cards of one comparison that differ in `gt.version`, `gt.signature`, `evaluator` hash or frame hash, unless a line in the card says why (today only the GT signature is compared).
- A card whose evaluator is not clean (or was not checked) is a blocker in `compare`; `unknown` on either side of a key is "cannot tell" and `compare` exits 2, never 0.
- Old cards keep what they have; missing fields read "unknown", never a guess.

## Reproduce a result: `repro.py <run id>` (one small tool, prints, changes nothing)
The code tag (`bf/<id>` or `lcr/<id>`), the autogt tag of the GT version (`autogt/vN`) and the rebuild command if the store is gone, the split and frame hash, the evaluation command from `run.json`, and what differs on this machine (python, torch).

## Who and when
The tools owner writes the collector and `repro.py` (`autoresearch/tools`, a selftest, a critic pass); the problem's evaluation writer (`bev_fusion/evaluation/bev_suite.py`, owner: BEV system chat) calls it; the GT chat has stamped `gt_version` into the index already. The training side (`gt_version` next to `gt_signature` in `gt.json`) waits for the freeze of `bev_fusion/training/*.py`; the evaluation side and the tools can start before.
