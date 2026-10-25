# training - code of the camera + radar BEV system

Plain PyTorch (env `gcar`: PyTorch 2.8, Python 3.12, no mmcv). Every module has a self-test with known answers (`python training/<module>.py`).

| file | what it does | self-test |
|---|---|---|
| `camera_encoder.py` | front-camera encoder with pre-trained camera heads: ResNet-50 + CustomFPN + DepthNet (camera-aware gate, 27-vector) with DualBEV's layer names (`load_pretrained` reports missing / unexpected / shape-mismatched keys), depth softmax over 118 bins (1-60 m) -> lift PD -> BEV (B, 128, 120, 104); `freeze_camera_heads` keeps backbone, neck, DepthNet frozen with BatchNorm in eval mode; `nuscenes_front_ida`, `camera_vector` | shapes, key names, image transform, one camera alone = in a batch, frozen heads get no gradient |
| `radar_bev.py` | radar block (simple): detections from any rig (position, distance, ray direction in the vehicle frame, Doppler, RCS, age) -> hard raster into the cell they reside in (count, mean properties with direction and Doppler as circular values, sub-cell offset) -> two 3x3 convs -> (B, 32, 120, 104) on the camera BEV grid; `to_vehicle` (sensor polar + pose -> vehicle frame), `doppler_code` | periodic Doppler code, sensor pose, cell and properties, mean of a cell, frames, rig independence, module |
| `lift_pd.py` | camera block: lookup table per calibration (voxel -> image position + fractional depth bin) and lift PD (pull with depth weight: feature x p(depth) by two grid_sample gathers), height into channels; option learned elevations; `use_depth=False` = lift P | geometry, uniform depth = P / D, one-hot depth, rig shift (camera 1 m forward = BEV shifted 2 cells), gradcheck incl. learned heights, module |
