# tests - synthetic gates of the blocks (plan.md section 10.1)

No block is chained into the next before its synthetic test with a known answer passes. Run from the bev_fusion folder with the `gcar` env, e.g. `wsl ~/miniconda3/envs/gcar/bin/python tests/<file>.py`.

| file | what it tests | result |
|---|---|---|
| `synthetic_world.py` | the shared synthetic world: one BEV scene (lane lines, boxes with heading / velocity / height / class, ego speed and yaw rate) -> ray-cast camera image + exact depth + pixel classes, radar detections (Doppler = v . u, RCS, noise, road-edge clutter), GT layers with a velocity field, the next frame with exact ego motion; own self-test | self-test passes (box and ground depth exact, Doppler, step, GT velocity) |
| `test_oracle_lift.py` | lift PD with the exact depth: do road, lane lines, vehicles land in the cells their rays hit? variants of height sampling and depth representation, stride 1 and 8; optional picture | 2026-10-25: Grid slices miss the road; histogram depth + vertical oversampling x4 recovers it to far range (details in plan.md 10.1) |
