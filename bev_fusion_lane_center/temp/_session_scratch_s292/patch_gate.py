import os


def patch(path, anchor, new):
    s = open(path, encoding="utf-8", newline="").read()
    assert s.count(anchor) == 1, (path, s.count(anchor))
    s = s.replace(anchor, anchor + new)
    tmp = path + ".tmp_gate"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(s)
    os.replace(tmp, path)


GATE = ('    if dev == "cuda":                                                                                                                          # D100: a dump / score next to two trainings crashed one of them (CUDA unknown error, 2026-11-14)\n'
        '        import gpu_gate\n'
        '        gpu_gate.wait_for_room(who="%s")\n')
patch("evaluation/jends_eval.py", '    dev = "cuda" if torch.cuda.is_available() else "cpu"\n', GATE % "jends_eval dump")
patch("evaluation/jmaps_eval.py", '    dev = "cuda" if torch.cuda.is_available() else "cpu"\n', GATE % "jmaps_eval score")
print("patched")
