p = "training/README.md"
s = open(p, encoding="utf-8").read()

def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a)
    s = s.replace(a, b)

# model.py row
rep("`check_config` accepts it with `camera_features` `dualbev` and `dual`, t20); `tests/test_motion_joint.py` |",
    "`check_config` accepts it with `camera_features` `dualbev` and `dual`, t20); `cam_buffer` (t21, default `window` = the path above, bit for bit): `pair` = the temporal block's window is ONE buffered frame (the previous raw camera BEV), `recurrent` = the previous frame's fused output is the buffer (each frame lifted once, fused with the previous fused state through the block's splat + velocity head, a batch unrolls its frames oldest to newest from the previous call's buffer or cold; the motion block reads the two RAW BEVs, current + the frame before it, in every mode but `window`), no new parameter; stream mode = a batch of ONE frame with the previous frame from `state[\"buf\"]` and `step` = the ego step since it (a triple or the sweep chain), `advance` = a stream frame that needs no output (lift + temporal block only); `tests/test_motion_joint.py`, `tests/test_stream_temporal.py` |")
rep("S5c training path (gradient to the refiner, trainer float cast) |",
    "S5c training path (gradient to the refiner, trainer float cast); `cam_buffer`: stream = batch to 1e-5 (pair, recurrent, across key frames), gradient through the carried fused state, cold start, `window` unchanged, 8 planted bugs killed (`../tests/test_stream_temporal.py`) |")
# temporal_fusion.py row
rep("Explicit data (radar Doppler vectors / tensors, counts) via `n_explicit`, `vec2`, `sym2`: rotated with the ego rotation, kept out of the memory |",
    "Explicit data (radar Doppler vectors / tensors, counts) via `n_explicit`, `vec2`, `sym2`: rotated with the ego rotation, kept out of the memory. `step` is one (dt, v, omega) triple or a chain of them (`as_chain`, `advance_chain`: the memory pose advances along the chain, the decay time is its sum; t21 stream mode) |")
# configs line
rep("`configs/` holds the training configurations (`python training/train.py configs/<name>.json`, base `t1_base.json`, arms in its README);",
    "`configs/` holds the training configurations (`python training/train.py configs/<name>.json`, base `t1_base.json`, arms in its README; t21 (D51): `t21_pair.json` / `t21_pair4.json` (stride 4) / `t21_rec.json` / `t21_ctl.json` = `t20_base.json` warm-started from `runs/t20_base_s0` for 2000 steps at lr 1e-4, differing in `model.cam_buffer` and `n_cam` only);")
open(p, "w", encoding="utf-8", newline="").write(s)

p = "tests/README.md"
s = open(p, encoding="utf-8").read()
rep("| `test_report_build.py` |", '''| `test_stream_temporal.py` | one buffered frame instead of the window, `ModelConfig.cam_buffer` 'pair' / 'recurrent' (card t21, D51; tiny cached model, CPU, no data, about 2 min). A: pair stream (ONE frame + `state["buf"]["raw"]` + the ego sweep chain as `step`) = the batch of the two frames to 1e-5 at dt -0.5 / 0 / +0.5, also with the motion block; the pair batch = the window model on the same two frames bit for bit. B: recurrent stream of three frames = the batch unroll; across key frames (the trainer's way: the previous batch's state and the keyframe gap step) = the stream over six frames; the carried buffer changes the output; a batch without a step never reads a buffer, a pair batch never; the motion field = the window model's on the last two frames. C: the loss reaches the oldest frame's lift through the carried fused state (not only through the world memory) and the first key frame's lift through the buffer carried to the next one. D: cold start finite and = the window model with one frame; `advance` gives the state of the full step; refusals. E: `window` (default and named) = the code before t21 (the old camera composition is in the file), state_dict keys / shapes unchanged (golden digest of test_lane_branch_dual), `check_config` rules. H: `train.run_clip` over two key frames carries the buffer. F: 8 planted bugs. G: one composed triple against the sweep chain on a turning ego (0.006 cells for a frame step, 0.16 for the keyframe step). Run: `wsl ~/miniconda3/envs/gcar/bin/python tests/test_stream_temporal.py` | all checks pass, every mutant killed | passes (2026-11-04, CPU, about 2 min): 8 / 8 killed (carry ignored, current frame as the previous one, raw / fused swapped in the buffer, step dropped, gap step not reduced to the oldest frame, a batch that reads a buffer, motion block on the oldest frame, carry cut by a detach) |
| `test_report_build.py` |''')
open(p, "w", encoding="utf-8", newline="").write(s)

p = "tests/run_gates.sh"
s = open(p, encoding="utf-8").read()
rep('''run "evaluate door" ''', '''run "stream temporal"  "$PY" tests/test_stream_temporal.py          # cam_buffer pair / recurrent (t21): stream = batch, buffer rules, gradient through the carry, window unchanged, 8 mutants; CPU, about 2 min, no data
run "evaluate door" ''')
open(p, "w", encoding="utf-8", newline="").write(s)
