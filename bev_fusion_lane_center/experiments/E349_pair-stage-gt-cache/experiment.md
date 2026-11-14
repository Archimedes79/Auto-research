# E349 - the per-frame ground truth of the pair stage computed once: a derived-data cache and a pair-only item path, bit-identical fields, so that every later pair / maps arm trains at the speed of the GPU instead of the loader

Card of 2026-11-14 23:50, written by the lane experiments chat on the lane lead's order (Mirko 2026-11-14 23:30: "warum ist die Auslastung so schlecht?"; D101: 4 data workers). Plan card: nothing is built, nothing is run. It is the lever that makes every later arm cheaper (E348, E318 / E319 follow-ups, the pruning step 1 of D99, Q-Lane heads on the pair stage). D86: simple (one cache, one flag), little compute (the same targets are no longer rebuilt about 3 - 6 times per run and again in every arm), E2E untouched (inputs identical).

## 0. The card (what the tools run)
```json card
{
 "id": "E349",
 "card": "",
 "title": "pair-stage / maps-stage loader with a per-frame cache of the deterministic ground truth (junction maps, ends, pairs, pair scales, connector polylines) and a pair-only item path, against the uncached loader with the same 4 workers: identical batch fields, loader seconds per batch, training step time",
 "kind": "variant",
 "deterministic": true,
 "metric": "loader_seconds_per_batch",
 "direction": "lower",
 "band": 0.1,
 "prediction": {"value": -1.6, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = the loader as it is (`LD.loader('train', bs=8, workers=4, shuffle=True, map_mode='train', pixels=False)` with the flags of `jends_train.py`: junction, pscale, ppts): about 0.97 s per item in one process (profile, `temp/loader/profile_items.txt`), 1.83 s per batch of 8 with 4 workers on the loaded box (`temp/loader/identity_w4.txt`), a training step of about 2.0 s"},
  "v": {"recipe": "the same loader with `QData(..., cache=<dir>)`: the item is the feature read plus a read of the cached fields, B's targets and the simulated map are not computed (the pair-only item path); `collate` stacks only the fields that exist"}
 },
 "control": "c",
 "variant": "v",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "loader_seconds_per_batch", "vs": "c", "min_effect": 0.9}
 ],
 "budget_gpu_h": 0.0,
 "kill": "a cached field differs from the freshly computed one in ANY frame checked (np.array_equal with NaN equal to NaN: not a tolerance), or a batch of the cached loader differs from the uncached one in a field the stage reads (the first 50 batches, both seeds), or the cache was built under another ground-truth signature or other code than the loader reads (refused by the loader)",
 "evaluate": {"split": "train"}
}
```
By hand (label "by hand: measurement"): adopted as the default of `jends_train.py` / `jmaps_train.py` (flag `--cache`, default on once adopted) when (a) the field identity holds on ALL frames the build verifies, (b) the batch identity holds for seeds 0 and 1, (c) the loader is at least 2 times faster per batch with 4 workers on the same box (min_effect 0.9 s of the 1.83 s), (d) the logged losses of the first 50 steps of a 200-step smoke with and without the cache agree to 1e-4 (GPU arithmetic is not bit-reproducible across runs; the inputs are), (e) a code critic pass. Nothing about any model changes; no TEST.

## 1. What the profile says and what the cache holds
- **Measured (2026-11-14 23:45, `temp/loader/profile_items.py`, 24 items, one process, cProfile, workers 0, CPU):** 0.97 s per item. By callee, per item: `fit_pair_scales` 0.39 s (the grid search of the fitted handle scales, `qjunction.py:36`), `junction_gt` 0.33 s (of which 0.29 s the shapely `buffer` calls that grow the connectors into the zone and core, `jgt.py:184`), **`fit_long_scales` 0.15 s (B's long-link scales: the pair stage never reads it)**, `node_table` 0.03 s and the remaining B targets, the lane GT load 0.02 s, `end_table` 0.015 s. With 3 workers a batch of 8 costs 2.4 - 2.5 s, with 4 workers 1.83 s: **the training step of about 2 s is the loader** (`experiments/E342_pair-head-host-time/experiment.md` section 4: batches are bit-identical for every worker count).
- **What the pair stage reads** (grep of `jends_train.py`, `jends.py`, `jmaps.py`): `x`, `ent`, `ext`, `ent_m`, `ext_m`, `pairs`, `pdrv`, `pscale`, `ppts` (only with `--geom-w`), `jz`, `jheat`, `jdir` and, for the maps loss, `jpaint`; it does NOT read B's targets (`exists`, `flags`, `offset`, `dirv`, `sstart`, `piece`, `uout`, `uin`, `tier`, `vis`, `ztier`, `lk`, `lk_long`, `lk_scale`) nor the simulated map (`map_*`, `kmask`, `piece_map`, `piece_road`, `tmask`: unseeded, E342 section 4). The cache holds exactly the deterministic fields it reads and that are expensive: `jz`, `jheat`, `jdir`, `jpaint`, `ent`, `ext`, `pairs`, `pdrv`, `pscale`, `ppts` (both with and without the optional ones: `pscale` and `ppts` are cached once, the flags choose what is returned). `x` (the BEV features, normalised, fp16) stays a read from the existing feature files (a memory-mapped row, no computation).
- **Size:** about 0.6 MB per frame (the junction maps dominate: `jpaint` 2 x 200 x 200, `jdir` 4 x 100 x 100, `jheat` 2 x 100 x 100, `jz` 100 x 100) x 8 414 TRAIN frames, about 5 GB, plus the 873 VAL frames (0.5 GB). Place: D: (325 GB free; C: has 36 GB free), reached through a junction `data/pair_cache` made with `mklink /J` (D42: big derived data on D:, a project reaches it through a link in its own `data/`; never `ln -s` in Git Bash, it copies). Not in git (derived data; D40: only `autoresearch/tools/package.py` may delete it; the recipe is the builder in git).
- **Cost to build:** 8 414 x 0.75 s (the cached part of the item) / 4 workers is about 30 minutes plus the VAL frames; one build serves every later arm (a 3000-step run uses each frame about 2.85 times, a 6000-step run 5.7 times: the uncached loader rebuilds the same targets every time).

## 2. Design
- **Builder** `training/data/prep/pair_cache.py` (promoted code: a test, a README row; the existing `data/prep/junction_paint.py` is the model): for every frame of a split it calls the SAME functions as `QData.__getitem__` (`QEND.end_table`, `QJ.junction_targets`, `QJ.fit_pair_scales`, `QJ.pair_polylines`) and writes one file per frame (an `.npz` of the fields, written atomically) into `data/pair_cache/<GT signature>/<split>/<token>.npz` plus a `manifest.json` (GT signature, the sha256 of the source files of the producing functions `qjunction.py`, `jgt.py`, `qends.py`, `qtargets.py`, the numpy / shapely versions, the field list and dtypes). Four workers; resumable (an existing file is kept after its `np.load` check); no overwrite of a file whose manifest differs (refused: rebuild into another folder).
- **Loader** `QData(..., cache=dir)` (lane_gt_data.py; default off, so every running and every stored command line is unchanged bit for bit): the pair-only item path reads `x` and the cached fields and skips `node_table`, `grid_targets`, `link_array`, `fit_long_scales`, `make_map`, `piece_targets` and the junction GT; `collate` stacks only the fields that exist (the B fields are `if n in b[0]` guarded now only for some of them: the pair-only path needs the guard for all). The loader refuses a cache whose manifest signature or code hashes differ from the current ones.
- **Trainers:** `jends_train.py` and `jmaps_train.py` get `--cache` (default off until adopted, then on); `jends_eval.dump` / `jmaps_eval` keep the uncached VAL loader (they are not the bottleneck and must stay the reference); the cache of VAL exists for the check only.
- **Controls (written before the build):** (1) FIELD IDENTITY: the builder recomputes every frame once more in a second pass on a fixed 5 % sample (seed 349) and the verification run compares ALL frames of VAL and 500 random TRAIN frames: `np.array_equal` (NaN equal to NaN), dtype and shape included; (2) BATCH IDENTITY: `temp/loader/loader_identity.py` extended with a `cache` run: the first 50 batches for seeds 0 and 1, cached against uncached, every field the stage reads equal; (3) SMOKE: `jends_train.py --smoke` (20 steps) is too short: a 200-step run with and without `--cache`, same seed, the logged losses of the first 50 steps equal to 1e-4; (4) the refusal tests: another signature, a changed code hash, a truncated file, a missing frame; (5) timing: the control and the variant measured in the same process under the same load, blocks A B | B A (as `training/ab_timing.py`), 40 batches each.
- **Prediction (numbers):** per item 0.97 s -> about 0.06 s (the feature read, 8 files per batch of the cache, collate); loader per batch with 4 workers 1.83 s -> about 0.2 s (-1.6 s, hit window +-0.3); the training step about 2.0 s -> about 0.8 s (the pair head's host work of E342 part A, 2.6 ms per frame, and the GPU become the limit; hit window 0.6 - 1.2 s; measured on the next arm that uses it, E348, not in this card); the build about 30 minutes for TRAIN.
- **Risks named:** the cache is bound to the GT signature (a GtGenerator edit changes it: a rebuild of 30 minutes, the loader refuses the old one); the cached `jpaint` depends on the paint sidecars (`junction_paint.py`): their hash enters the manifest; a bug in the cached path would change every later arm silently: the identity checks are the guard, run once per build and once per new code hash; B's training (unseeded simulated map) is not touched.
- **Not claimed:** anything about the model, TEST, or B's loader; the 0.8 s step (measured later); that the cache helps the dumps.

## 3. Referee
A fresh `ar-critic` (cards) on this card, then a coder for the builder and the loader path with a test, then a `code` critic pass (the loader is shared code that running chains import: the change is default-off and atomic), then the build, the identity checks and the timing; the result goes into this card.
