def sub(path, old, new):
    s = open(path, encoding="utf-8").read()
    assert s.count(old) == 1, (path, old[:60], s.count(old))
    open(path, "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))

row = ("| `test_student_chain.py` | the camera student in the chain (t22 stage B: `ModelConfig.camera_student` / `student_init` / `student_trainable` / `student_frames`, the cached model's frame-0 splice, the anchor of `train.forward_loss`; CPU, 2 threads, about 30 s, 1.5 GB, no data, no ImageNet weights: a random ResNet-18 student and a stub registered in `model.STUDENTS`). "
       "A OFF is the code before: camera BEVs and the whole forward equal a copy of the old `camera_bevs` bit for bit (dualbev, bevfusion, dual, cam_adapter, camera_off), no `student` module / key / `student_side`. "
       "B splice: BEV = the student-free model's on the batch whose frame 0 was replaced by the student's outputs (stub = function of the image, F = 1, 2, 3), frames >= 1 stay the cached teacher's, cached garbage in frame 0 changes nothing, a missing `lane_img` is a KeyError naming `lane_images`. "
       "C `student_side` (student and detached teacher tensors of frame 0, taken and cleared by `_forward`). "
       "D the real student: `student_init` loads the newest checkpoint's EMA tensors strictly; trainable: every one of the 80 student tensors gets a gradient, BatchNorm statistics fixed under `model.train()` / `ema.applied` (a plain encoder's move: control); frozen: no gradient, no graph, no `student.` key in the checkpoint state; built last (all other tensors equal at one seed); `lr_groups` give the student its own rate; warm start keeps the loaded student; a chain checkpoint without a student loads into the chain with a frozen one (`bev_adapter.load_run`) and is refused by a trainable one. "
       "E config refusals one by one (dualbev / bevfusion, lift 's', cam_adapter, camera_off, unknown kind / `student_frames`, `student_frames` 'all' = NotImplementedError, frozen without init, fields without the option, the live-camera model; `lane_branch` stays allowed), `train.build` refusals (cache, `lane_images`, anchor without a trainable student, schedule range). "
       "F anchor: 0 for student = teacher, by-hand positive otherwise, schedule 2.0 / 1.1 / 0.2 at 0 / 0.25 / 0.5+, `forward_loss` adds weight x anchor in training (gradient reaches the student) and only logs it in validation, `run_clip` logs it. "
       "G the camera-off class of bug: `bev_modules.zero_cam`, zeroing or swapping only `lane_img` and the shuffle control change the camera BEV of a model with the student "
       "| all checks pass | passes (2026-11-05, CPU, 26 s); 14 planted bugs in a copy of the code are killed (`temp/t22/mutate_chain.py`) |\n")
sub("tests/README.md",
    "| `test_student_encoder.py` |", row + "| `test_student_encoder.py` |")
sub("tests/run_gates.sh",
    'run "evaluate door"     "$PY" tests/test_evaluate_door.py',
    'run "student chain"    "$PY" tests/test_student_chain.py            # t22 stage B: camera_student in the chain: off = old code, frame-0 splice, student_side, trainable / frozen student, anchor, refusals; CPU, about 30 s, no data\nrun "evaluate door"     "$PY" tests/test_evaluate_door.py')
