# training - code of the camera + radar BEV system

Plain PyTorch (env `gcar`: PyTorch 2.8, Python 3.12, no mmcv). Every module has a self-test with known answers (`python training/<module>.py`).

| file | what it does | self-test |
|---|---|---|
| `lift_pd.py` | camera block: lookup table per calibration (voxel -> image position + fractional depth bin) and lift PD (pull with depth weight: feature x p(depth) by two grid_sample gathers), height into channels; option learned elevations; `use_depth=False` = lift P | geometry, uniform depth = P / D, one-hot depth, rig shift (camera 1 m forward = BEV shifted 2 cells), gradcheck incl. learned heights, module |
