R = "/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/"
p = R + "jprune_probe.py"
s = open(p, encoding="utf-8").read()
a = s.index('    wsl bash tools/env/_py14.sh jprune_probe.py sweep')
b = s.index('    wsl bash tools/env/_py14.sh jprune_probe.py time')
s = s[:a] + '''    wsl bash tools/env/_py14.sh jprune_probe.py sweep  --run e312_p_s0 --out temp/e345/e312_p_s0 [--score-points f.json] [--runs-dir DIR] [--max N] [--val-stride N] [--calib-max N] [--workers N] [--device cpu] [--deterministic] [--no-controls]
                                                                                                 # resumes from sweep.json; the calibration is kept in calib.pt (a rerun picks the same channels); relative paths are relative to the problem folder
''' + s[b:]
old = '''`dump_frames` is `jends_eval.dump` without its file writing (tests/test_jprune_probe.py holds them equal). The point sets named in `--score-points` also get `runs/e345_p<NNN>_<model>/ends_val.pkl` + a symlink to the unmodified model.pt, which `jprune_table.py score` scores.
Choices the card left open: the TRAIN split is called "train" (the loader has no "train_clean"); a tap with no valid position in the calibration (no keypoint on any frame) gets mean 0, s_c 0 and is then allowed to be unchanged; the two nulls are compared by ends_mean6
(cap 3e-3) here, by the scored cards in `jprune_table.py`.
"""'''
new = '''`dump_frames` is `jends_eval.dump` without its file writing, but NOT yet shared with it (TODO: `jends_eval.dump` should call it once no chain runs `jends_eval.py`): the sweep and the point dumps use `forward_batches` + `frames_of` (the pieces `dump_frames` is made of), and
tests/test_jprune_probe.py compares `dump_frames` frame for frame with a fresh `jends_eval.dump` on 16 frames. The point sets named in `--score-points` also get `runs/e345_p<NNN>_<model>/ends_val.pkl` + `point.json` (the set, the edits, the sha256 of the pickle, written LAST: a folder without it
is incomplete and redone on resume) + a symlink to the unmodified model.pt, which `jprune_table.py score` scores; every file is written through a temporary file and `os.replace`.
Choices the card left open: the TRAIN split is called "train" (the loader has no "train_clean"); a tap with no valid position in the calibration (no keypoint on any frame) gets mean 0, s_c 0 and is then allowed to be unchanged; the two nulls are compared by ends_mean6
(cap 3e-3) here, by the scored cards in `jprune_table.py`; the random draws of the control use RandomState(1000 * draw + sum(ord(c) for c in unit name)) for draw 0 / 1 / 2; the attention-head edit is checked against the by-hand per-head replacement to 2e-7 (measured 6e-8 and 9e-8; the card asks 1e-7).
GPU safety: before any CUDA work `wait_for_gpu` calls `gpu_gate.wait_for_room` (D100) with its own, much longer timeout (12 h, `E345_GATE_TIMEOUT_S`; the shared `GPU_GATE_TIMEOUT_S` of the chains is ignored here) and, if the GPU is still full afterwards, exits with code 75 and a message instead of going on;
`time` and every pass record of `sweep.json` carry `torch.cuda.max_memory_allocated / reserved` (MiB). Determinism: `--deterministic` sets `CUBLAS_WORKSPACE_CONFIG`, cudnn.deterministic and `use_deterministic_algorithms` BEFORE CUDA is up and is recorded in `sweep.json`; a resumed job takes the recorded setting
(and refuses `--deterministic` against passes made without it); an identity hook that does not reproduce the null to 0 means: restart with `--deterministic`.
"""'''
assert old in s
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
p = R + "jprune_cost.py"
s = open(p, encoding="utf-8").read()
old = 'PREFIX = (("jmodel.pyr.", "pyr."), ("jmodel.jm.", "jm."), ("je.", "je."))\n'
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, ""))
print("ok")
