def rd(p):
    return open(p, encoding="utf-8").read()
def wr(p, s):
    open(p + ".new_s292", "w", encoding="utf-8", newline="\n").write(s)
def sub(s, a, b):
    assert s.count(a) == 1, a[:80]
    return s.replace(a, b)

q = rd("evaluation/qends_eval.py")
q = sub(q, "import jwire as JW  # noqa: E402\n", "import jwire as JW  # noqa: E402\nimport jwire_head as JH  # noqa: E402\n")
q = sub(q, '    assert wire in (None, "twin") and (wire is None or (bool(getattr(a, "snap", False)) and not getattr(a, "snap_all", False) and not getattr(a, "snap_guard", False) and getattr(a, "attach_r", None) is None)), "--wire twin needs --snap, no --attach-r, no --snap-all / --snap-guard"\n',
          '    assert wire in (None, "twin", "head") and (wire is None or (bool(getattr(a, "snap", False)) and not getattr(a, "snap_all", False) and not getattr(a, "snap_guard", False) and getattr(a, "attach_r", None) is None)), "--wire twin / head needs --snap, no --attach-r, no --snap-all / --snap-guard"\n'
          '    wire_run = getattr(a, "wire_run", None)\n'
          '    assert (wire == "head") == bool(wire_run), "--wire head needs --wire-run <run of the head> (and --wire-run only goes with --wire head)"\n'
          '    head = JH.load_wire(wire_run) if wire == "head" else None                                                              # E325: the learned head (runs/<wire_run>/model.pt)\n')
q = sub(q, '''            wc = JW.twin_choice(JW.kept_ends(graph.lines, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), pairs, cover_pos())[1], pairs, WIRE_POS, WIRE_ANG)
''', '''            wc = JW.twin_choice(JW.kept_ends(graph.lines, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), pairs, cover_pos())[1], pairs, WIRE_POS, WIRE_ANG)
        elif wire == "head":                                                                                                    # E325 arm w: the learned head chooses the pieces (its features never read the GT)
            keep_, ends_ = JW.kept_ends(graph.lines, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), pairs, cover_pos())
            attrs_ = JH.piece_attrs(graph.lines, bk["kind"], bk["piece_nodes"], bk["piece_links"], np.asarray(pred["nodes"]["e"], float).reshape(-1), np.asarray(pred["links"]["p"], float).reshape(-1), keep_)
            wc = JH.wire_choice_head(head, ends_, attrs_, pairs, [(float(e["kp_p"][i]), float(e["kp_p"][k + j])) for i, j in zip(ia, ib)], WIRE_POS)
''')
q = sub(q, '    out["wire"] = wire\n', '    out["wire"] = wire\n    out["wire_run"] = wire_run\n')
q = sub(q, '    s.add_argument("--wire", default=None, choices=(None, "twin"), help="E325: choose the wired pieces through add_connectors(wire_choice=...): twin = the zero-parameter twin of the nearest-piece rule (needs --snap and --tag); must reproduce the E312 cards")\n',
        '    s.add_argument("--wire", default=None, choices=(None, "twin", "head"), help="E325: choose the wired pieces through add_connectors(wire_choice=...): twin = the zero-parameter twin of the nearest-piece rule, head = the learned head of --wire-run (needs --snap and --tag); the twin must reproduce the E312 cards")\n'
        '    s.add_argument("--wire-run", default=None, help="E325: the run of the head for --wire head (runs/<name>/model.pt, written by jwire_train.py run)")\n')
wr("evaluation/qends_eval.py", q)

j = rd("evaluation/jends_eval.py")
j = sub(j, 'wire=getattr(a, "wire", None), skip_base_check=True,', 'wire=getattr(a, "wire", None), wire_run=getattr(a, "wire_run", None), skip_base_check=True,')
j = sub(j, '            s.add_argument("--wire", default=None, choices=(None, "twin"), help="E325: the zero-parameter twin of the nearest-piece rule chooses the wired pieces (needs --snap and --tag)")\n',
        '            s.add_argument("--wire", default=None, choices=(None, "twin", "head"), help="E325: the zero-parameter twin of the nearest-piece rule (twin) or the learned head of --wire-run (head) chooses the wired pieces (needs --snap and --tag)")\n'
        '            s.add_argument("--wire-run", default=None, help="E325: the run of the head for --wire head")\n')
wr("evaluation/jends_eval.py", j)
print("ok")
