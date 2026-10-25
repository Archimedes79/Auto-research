# tests - synthetic gates of the blocks (plan.md section 10.1)

No block is chained into the next before its synthetic test with a known answer passes; each gate script checks its own pass criterion and exits with 0 (pass) or 1 (fail). Run all from the bev_fusion folder: `wsl bash tests/run_gates.sh` (fast self-tests and gates) or `wsl bash tests/run_gates.sh --all` (also the training gates, minutes on the CPU). The real-data counterpart of every gate is listed in plan.md section 10.2.

| file | what it tests | gate criterion | result 2026-10-25 |
|---|---|---|---|
| `run_gates.sh` | runs every self-test and gate, one PASS / FAIL line per block | - | all fast gates pass |
| `synthetic_world.py` | the shared synthetic world: one BEV scene (lane lines, boxes with heading / velocity / height / class, ego speed and yaw rate) -> ray-cast camera image + exact depth + pixel classes, radar detections (Doppler = v . u + noise, RCS, road-edge clutter), GT layers with a velocity field, the next frame with exact ego motion | own self-test (box and ground depth exact, Doppler, step, GT velocity) | passes |
| `test_oracle_lift.py` | lift PD with the exact depth: road, lane lines, vehicles in the cells their rays hit; variants of height sampling and depth representation, stride 1 and 8; optional picture | recommended lift (oversampling x4 + histogram depth, stride 8): lanes >= 0.9, vehicles >= 0.9, lanes at 40-60 m >= 0.7 | passes (0.99 / 1.00 / 0.81); Grid slices without oversampling miss the road |
| `test_doppler.py` | full 2D velocity of moving boxes from Doppler via the information form A, b: 1 radar 1 sweep, 3 radars, 1 radar 6 sweeps | radial median <= 0.3 m/s everywhere; tangential median < 1 m/s with 6 sweeps | passes (radial 0.02-0.05; tangential 1 sweep 1.7-2.4, 6 sweeps 0.18-0.57 m/s; perfect association assumed) |
| `test_flow_from_frames.py` | velocity field from two consecutive BEV frames without a velocity channel: (a) supervised, (b) self-supervised (forward splat must match the next frame) | per-box median error < half of the null baseline, for (a) and (b) | (a) passes (0.94 vs null 6.05 m/s); (b) FAILS (4.88), open |
