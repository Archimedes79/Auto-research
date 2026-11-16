def sub(p, a, b):
    s = open(p, encoding="utf-8").read()
    assert s.count(a) == 1, (p, a[:70])
    open(p + ".new_s292", "w", encoding="utf-8", newline="\n").write(s.replace(a, b))
sub("evaluation/qends_eval.py", '    head = JH.load_wire(wire_run) if wire == "head" else None                                                              # E325: the learned head (runs/<wire_run>/model.pt)\n',
    '    head = JH.load_wire(*wire_run.partition(":")[::2]) if wire == "head" else None                                          # E325: the learned head `runs/<run>/model.pt`, or `<run>:<file>` for another checkpoint of the run (model_h1.pt: a second head seed)\n')
