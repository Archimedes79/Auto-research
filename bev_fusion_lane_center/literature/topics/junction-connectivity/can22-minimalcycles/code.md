# can22-minimalcycles: what the code offers

Status: checked, read through the page fetch tool on 2026-11-10 (extracts of the pages; nothing cloned, nothing saved). Repository: https://github.com/ybarancan/TopologicalLaneGraph

## Facts of the repository page and API
- Licence: none. GitHub API `license` is null, the README names none; not stated for the linked STSU, Polygon-RNN, DETR, PINET either. Treat the code as not reusable without asking the author (we would write our own port in any case).
- 80 stars, 9 forks, 6 open issues (not read), 7 commits, last push 2023-01-29, created 2022-03-15 (https://api.github.com/repos/ybarancan/TopologicalLaneGraph). Python.
- Data hosted on data.vision.ee.ethz.ch (README): checkpoints `PRNN_MC_Nusc.pth`, `TR_MC_Nusc.pth`, `PRNN_MC_Argo.pth`, `TR_MC_Argo.pth`; extracted cycles `nusc-polygon-dicts.zip`, `argo-polygon-dicts.zip`. Not downloaded by us.

## Structure
- root: `train_tr_nusc.py`, `train_tr_argo.py` (transformer), `train_prnn_nusc.py`, `train_prnn_argo.py` (Polygon-RNN), `validator_*` (four test scripts), `poly_gt_saver.py` (driver that runs the dataset in `gt_polygon_extraction` mode and saves `[blob_mat, poly_centers, poly_one_hots, blob_ids]` per sample into one .npy dict; batch size 1, only element 0 of a batch is stored).
- `configs/`: `defaults.yml`, `defaults_argo.yml`, `polyline_defaults.yml`, `polyline_defaults_argo.yml` (hard-coded cluster paths, e.g. `/cluster/work/cvl/cany/...`).
- `scripts/`: `make_nuscenes_labels.py`, `make_argoverse_labels.py` (README step 2 names `make_labels.py`, identical to STSU).
- `src/detr/`: `detr.py` (model with pair head and cycle heads, `SetCriterion`, `build`), `matcher.py` (Hungarian for curves and cycles, `min_max` re-labelling), `deeplab_backbone.py`, `polyline.py`, `polyline_polygon.py`, `transformer.py`, `position_encoding.py`.
- `src/data/`: `data_factory.py`, `nuscenes/dataset.py` (`NuScenesMapDataset`), argoverse, augmentation helpers.
- `src/utils/`: `bezier.py`, `configs.py`, `confusion.py` (metrics), `geometry.py`, `visualise*.py`. Also `src/nn/` (pyramid, resampler, resnet, transformer, losses) and `src/deeplab/`.

## Entry points
`train_tr_nusc.py` (model from `src.detr.detr.build`, data from `data_factory.build_nuscenes_dataloader`), `validator_tr_nusc.py`, `poly_gt_saver.py --dataset_name nuscenes --interval_start --interval_end`. The README gives no command lines.

## What is reusable for us
- Reading value, not import value: `DETR.get_assoc_estimates` / `thresh_and_assoc_estimates` (pair grid), `SetCriterion.loss_assoc` (weighted BCE + `loss_end_match`), `loss_poly_hamming` + minimality hinge, `matcher.min_max` (how GT cycles are re-labelled to predicted curves).
- Not reusable: the dataset class (hard-wired to label PNGs of STSU and the nuScenes map API token order), the metrics (`confusion.py`), the query decoder.

## State and bugs seen (extracts, one read each)
- dataset.py: polygon targets (`blob_mat`, `poly_one_hots`, ...) are placeholders; the graph-cycle extraction of the paper is not visible there; the cycle ground truth is meant to be downloaded.
- confusion.py: MC-F and H-GT-F are not named in the code; the common-polygon and RNN-order blocks are commented out; `confusion.reset` is called without parentheses in train_tr_nusc.py; `iou_list` is never filled (miou logs nan).
- detr.py: `assoc_net=model.association_classifier` is read after the DataParallel wrap (fails on several GPUs); `forward` returns `assoc_features`, not `pred_assoc` (inference needs `thresh_and_assoc_estimates` called explicitly); membership uses `est > 0.5` on logits in one place and `sigmoid < 0.5` in another.
- matcher.py: the `val` branch never sets `poly_total_return` (NameError caught, returns None); 105 polygon slots and 5 extra slots hard-coded.
- Net: the code documents the method well enough to re-implement the pair head and the loss terms; it will not reproduce Table 1 without repairs and the author's label files.
