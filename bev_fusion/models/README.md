# models - weights used by bev_fusion (not in git)

| file | what | source | check |
|---|---|---|---|
| `pretrained/dualbev-r50-cbgs.pth` | DualBEV single-frame R50 detector (nuScenes, EMA weights); we take only `img_backbone.`, `img_neck.`, `img_view_transformer.depth_net.` (428 tensors) for `training/camera_encoder.py` | DualBEV authors' Google Drive folder (file id 1rUM1yQzQKRd0GH1LgOtpN5p8oqQu0L2a), downloaded 2026-10-25 with Mirko's OK by `temp/setup/get_dualbev_ckpt.sh`; code Apache-2.0, no licence stated for the weights (trained on nuScenes: non-commercial) | 332,081,449 bytes, sha256 in `pretrained/dualbev-r50-cbgs.pth.sha256`; loaded with `torch.load(weights_only=True)` (tensors only); `temp/setup/check_dualbev_load.py`: 428 loaded, 0 missing, 0 unexpected, 0 shape mismatches, expected depth per image row 36 m near the horizon to 4.8 m at the bottom, mean peak probability 0.34 |
