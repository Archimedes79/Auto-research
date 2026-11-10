def sub(path, old, new):
    s = open(path, encoding="utf-8").read()
    assert s.count(old) == 1, (path, old[:60], s.count(old))
    open(path, "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))
sub("tests/README.md",
    "| all checks pass | passes (2026-11-05, CPU, 26 s); 14 planted bugs in a copy of the code are killed (`temp/t22/mutate_chain.py`) |",
    "| all checks pass | passes (2026-11-05, CPU, 26 s); every one of 35 planted bugs in a copy of the code is killed (wrong or all frames replaced, sources swapped, student on the wrong sample's image / ignoring the image, teacher not detached or from the wrong frame, option ignored, BatchNorm in train mode, anchor sign / schedule (3 ways) / logging / validation total, resume switch, `_external` (2 ways), checkpoint loading (oldest, live state, not strict), five `check_config` / `build` holes, frozen student with grad, `student_side` not set / cleared / returned; scratch `temp/t22/mutate_chain.py`), and the 7 stage-A mutants whose code moved to `student_encoder.py` (loss, eval-only loader; `temp/t22/mutate_moved.py`) |")
sub("tests/README.md",
    "plus 11 combinations (dual + ego-GRU,",
    "plus 12 combinations (the t22 camera student, a random ResNet-18 that trains in the 'dual' chain: gradient to every one of its tensors; dual + ego-GRU,")
