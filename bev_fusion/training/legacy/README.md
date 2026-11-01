# training/legacy - modules off the whole chain (moved 2026-10-31)

Nothing of the live chain imports these modules: not `../model.py` / `../train.py` and their imports, not `../../evaluation/`
(checked with a search of every import on 2026-10-31, repository review `../../../autoresearch/history/repo_review_2026-10-31.md`
section 7, Mirko's decision D24). They stay runnable: each puts `training/` on `sys.path` and imports the live blocks from there;
the tests that use them add `training/legacy` to their path. Old run snapshots (`runs/<name>/code/`) keep their own copies.

| file | what it is | used by | self-test |
|---|---|---|---|
| `s1_camera.py` | S1 camera-only BEV segmentation (cached image heads -> lift PD / P / S -> trunk -> semantic head) with the published presets (`ours`, `dualbev`, `simplebev`, `lss`) | `../../tests/test_s1_pipeline.py` (card `../../experiments/s1a_lift_pd_vs_p/`), `../../temp/timing/time_s1_chain.py`, `../../plan/requirements.md` R-METH-3 | `python training/legacy/s1_camera.py` |
| `train_s1.py` | the S1 driver for synthetic and real batches (loss, schedule, per-scene SegMeters, null baseline, bootstrap) | `../../tests/test_s1_pipeline.py` | `python training/legacy/train_s1.py` |
| `camera_decoder.py` | camera-only decoder (lift PD, semantic + object head, depth) | `../../tests/test_camera_only.py`, `../../temp/checks/probe_cam_lanes.py` | `python training/legacy/camera_decoder.py` |
| `radar_decoder.py` | radar-only decoder (stationary map + moving objects, detections moved once to the output time) | `../../tests/test_radar_only.py` | `python training/legacy/radar_decoder.py` |
| `dualbev_front.py` | pre-trained DualBEV ported for the front-area SOA comparison (experiment t4) | `../../temp/t4/dualbev_front_eval.py`, `debug_channels.py` (card `../../experiments/t4_soa_front/`) | `python training/legacy/dualbev_front.py` (needs `models/pretrained/dualbev-r50-cbgs.pth`) |
| `light_encoder.py` | light stride-8 image encoder (ResNet-18 / EfficientNet-B0 / B2 + FPN + depth and context heads), plan 2.7 | `../../temp/timing/time_s1_chain.py`; its distillation script crashed and is in `../../_to_delete/off_chain_2026-10-31/` | `python training/legacy/light_encoder.py` |

`../temporal.py` is off the chain as well but stays in `training/`: the running queue `temp/checks/sup/queue_fused.sh` (tests/test_flow_fused.py) and nine tests import it from there; move it here when that queue has ended.
