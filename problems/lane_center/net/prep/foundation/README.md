# foundation - the GaussianCaR features, made again

Module 1 of `../../../DESIGN.md`: the frozen GaussianCaR (camera + radar BEV map segmentation, ICRA 2026) is run once on every frame and its
output is cached: 39 channels at 200 x 200 cells of 0.5 m, float16 = `dec32` (its decoder output, 256 channels, projected to 32 by a fixed PCA)
+ `logit7` (its seven map logits). The lane code reads the cache only (`<data>/lane_center/b123/feats/{train,val}_{dec32,logit7}.npy`, 30 GB);
this folder makes it again and says how close a new extraction is.

| file | what |
|---|---|
| `gcar_features.py` | the one command: the frozen network, the frames in the lane's frame, the two channel groups, the writer |
| `check_features.py` | a few frames extracted again and compared with the stored arrays |
| `pca_dec32.json` | the PCA basis of `dec32` (mean 256, basis 256 x 32, float32 by its sha256) |
| `b123_samples.json` | the nuScenes sample token of every row: train 8,414, val 1,706 (the part "val" = VAL 873 + TEST 833 of `data/splits.json`) |
| `build_rasteriser.sh` | builds the Gaussian rasteriser of the clone in place |
| `../../tests/test_foundation.py` | the test (no GPU, no weights, no data) |

## What must exist outside the repository
| piece | exactly | check |
|---|---|---|
| the foundation's code | `git clone https://github.com/santimontiel/gaussiancar <dir>`, commit `c0e748b9e95a6a6062579ddebdcd1c1e79500c2e` (2026-10-07); named by `--clone` or `GCAR_REPO` | the commit is compared; another one is refused |
| its rasteriser | `conda activate gcar && bash build_rasteriser.sh <dir>`: adds `#include <cstdint>` to six headers of `cuda_rasterizer/` (the only change of the clone, for gcc 13) and builds `_C*.so` in place | imported before the network is loaded |
| its weights | `nuscenes_map.ckpt`, 1,130,353,854 bytes, sha256 `a184b022ef6d73fcf3abd075176273912a0e1fbbdf589e284151f1e388fb9baa`, from `https://huggingface.co/santimontieleu/gaussiancar/resolve/main/nuscenes_map.ckpt` (Apache-2.0); named by `--weights` or `GCAR_WEIGHTS` | size and sha256; all 1,654 tensors load |
| raw nuScenes | v1.0-trainval: tables, six cameras, radar key frames and sweeps, at `<data root>/nuscenes` (`../../../../../datasets/nuscenes/README.md`); no lidar | folders looked for |
| the environment | `gcar` (`../../../../../autoresearch/env/`): it holds nvcc 12.9 and gcc 13 for the build | |

Flash attention is not installed: the point transformer of the network runs without it (the same weights, attention in fp32).
On this PC (2026-11-28): the clone at `~/work/gaussiancar` (that commit, the six header lines as its only change), the weights at
`~/work/gaussiancar_ckpt/nuscenes_map.ckpt` (hash checked). `build_rasteriser.sh` was run on a scratch copy of the commit's unchanged source
(5 minutes with 2 jobs); three TRAIN rows extracted with that build are as close to the stored ones as the table below says.

## Use (under WSL, from the problem folder)
```
export GCAR_REPO=<clone> GCAR_WEIGHTS=<nuscenes_map.ckpt>
python net/prep/foundation/gcar_features.py --out <new folder>      # val, then train: 10,120 frames, 30 GB, the token lists
python net/prep/foundation/check_features.py --n 8                  # 8 frames per part, twice, against the stored arrays
```
`gcar_features.py` never overwrites an array: it writes into a new folder, goes on with an interrupted part from its mark
(`<part>.extracting`) and refuses a finished one. About 0.4 s per frame on the RTX 3060 alone (record of the first extraction). The lane code
reads `<data root>/lane_center/b123/feats`; a new folder becomes that only by the owner's hand.
`check_features.py` reads rows of TRAIN and VAL only (the TEST rows of the part "val" stay closed) and writes
`<runs root>/lane_center/_scratch/foundation/check_features.json`.

## How close a new extraction is (measured 2026-11-28, RTX 3060, torch 2.8.0, `check_features.py --n 22`: 22 VAL + 22 TRAIN frames)
No frame is bit-equal, not even two extractions in a row: the rasteriser is not deterministic. Mean absolute difference per channel group
(mean absolute value: dec32 0.8, logit7 7.5) and the correlation per frame, median / worst:

| rows | group | new against stored | new against new (the noise) |
|---|---|---|---|
| VAL (22) | dec32 | 0.031, 0.9929 / 0.9703 | 0.020, 0.9976 / 0.9894 |
| VAL (22) | logit7 | 0.086, 0.9992 / 0.9959 | 0.057, 0.9997 / 0.9985 |
| TRAIN even rows (11) | dec32 | 0.022, 0.9986 / 0.9798 | 0.016, 0.9988 / 0.9972 |
| TRAIN even rows (11) | logit7 | 0.066, 0.9997 / 0.9983 | 0.046, 0.9997 / 0.9996 |
| TRAIN odd rows (11) | dec32 | 0.018, 0.9986 / 0.9967 | 0.017, 0.9985 / 0.9904 |
| TRAIN odd rows (11) | logit7 | 0.052, 0.9998 / 0.9996 | 0.050, 0.9998 / 0.9992 |

Single cells differ by much more (largest dec32 difference 46 against stored, 28 between two new runs). The stored odd TRAIN rows are a
new extraction's distance from itself; the even TRAIN rows and VAL are about 1.5 times as far. That fits the record of 2026-10-25 (tag `v1`,
`bev_fusion_lane_center/experiment_execution/plans/data_extension_b456.md`): the arrays were written on 2026-10-13 / 14 in the order VAL and
even TRAIN rows, then odd TRAIN rows, and the extraction code was rewritten in between; the code here is the later version, the earlier one
was never in git. What the difference does to the champion's numbers is not measured: the stored arrays stay the reference the networks were
trained and scored on, and they are kept.

## What cannot be made again
- The PCA basis was fitted on 2026-10-13 (the file's date; 300 frames by `gcar_pipeline1.sh` of the tag `v1`) by a command that was never
  in git. `pca_dec32.json` holds its numbers, bit for bit those of
  `<data>/lane_center/b123/pca_gcar.npz` (`dec_mean`, `dec_basis`; file sha256 `ba892f62294a51d67951147cde897127f2dd9e19b081c4e734fd4f14d5a96fda`).
- The stored arrays themselves (above). The other files of `b123` (the graph ground truth, the poses) are not made here; their scripts are in
  the tag `v1` (`bev_fusion_lane_center/training/data/prep/make_graph_gt.py`, `make_poses.py`).

## Left in the tag `v1` (`bev_fusion_lane_center/`)
| there | why not here |
|---|---|
| `training/data/prep/gcar_tools.py`: the group `fused32`, `probe` (the layout found against the ground truth), the sets `mini`, `b456`, `chk` | the stored set holds `dec32` + `logit7` of b123 only; the layout is a fixed turn here |
| `gcar_wide.py`, `gcar_pert.py`, `run_b456_extract.sh` | other feature sets (a wider window, perturbed inputs, nuScenes parts 4 - 6); none is stored or read by the champion |
| `gcar_pipeline1.sh`, `gcar_pipeline2.sh`, `run_gcar.sh` | the order and the launcher of the first extraction; one command here |
| `extract_features.py`, `crn_extract_radar.sh` | the BEVFusion features of loops 1 - 14 and the unpacking of radar archives (now `datasets/nuscenes/tools/`) |
| `tools/env/setup/setup_gcar_env.sh`, `gcar_nvcc.sh` | the environment step by step; now the export `autoresearch/env/environment_gcar.yml` |
| `tools/env/setup/gcar_patch_rast.sh`, `gcar_build_rast.sh`, `foundation_code.md` | here: `build_rasteriser.sh` and the table above (the other clones of that page, CRN and BEVFusion, are not used) |
