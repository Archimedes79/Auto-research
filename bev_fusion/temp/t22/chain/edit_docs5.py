def sub(path, old, new):
    s = open(path, encoding="utf-8").read()
    assert s.count(old) == 1, (path, old[:60], s.count(old))
    open(path, "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))
sub("tests/README.md", "passes (2026-11-05, CPU, 26 s); every one of 35 planted bugs", "passes (2026-11-05, CPU, 36 s); every one of 39 planted bugs")
sub("tests/README.md", "`student_side` not set / cleared / returned; scratch", "`student_side` not set / cleared / returned, the lane branch on the cached depth / the wrong frame / the wrong head / cut from the student's gradient; scratch")
sub("tests/README.md", "the camera student in the chain (t22 stage B:", "the camera student in the chain (t22 stage B, lane branch included:")
