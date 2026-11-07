def sub(path, old, new):
    s = open(path, encoding="utf-8").read()
    assert s.count(old) == 1, (path, old[:60], s.count(old))
    open(path, "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))

sub("training/README.md",
    "built last, so every other tensor's init is unchanged); the file also holds",
    "built last, so every other tensor's init is unchanged; for the current frame it replaces the DualBEV depth and context, the BEVFusion depth and context AND the lane branch's depth, older frames keep their cached outputs; add `student` to `grad_clip_split` when it trains. Know: the EMA copy of a frozen student drifts by about 3e-7 and `warm_start` / the evaluation apply that copy, not `student_init`; `zero_cam` gives the response to a black image; with `seq_bptt` the graphs of the student of every key frame of a clip stay in memory (not measured)); the file also holds")
sub("tests/README.md",
    "F anchor: 0 for student = teacher,",
    "H the lane branch lifts with the STUDENT's DualBEV depth of frame 0 (its contribution changes when only the student's p16 changes, not when only the cached prob does; the gradient reaches a trainable student; option off: the cached depth, bit for bit). I `grad_clip_split` 'student' names exactly the student's trainable tensors (a frozen one: refused), a huge student gradient is clipped on its own and leaves the other gradients as they were (the global clip scales them down). F anchor: 0 for student = teacher,")
