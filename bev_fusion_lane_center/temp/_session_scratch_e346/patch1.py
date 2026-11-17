p = "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e346/e346_lib.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)

rep('''def kind_labeller(arm, lane, gt):
    """kinds of delivered points: g0 = the old GT's own piece kinds, else the kind of the nearest lane GT point (its segment)"""
    if arm == "g0":
        pk = np.asarray([k == "connector" for k in gt["kind"]], int)
        return lambda P, I: pk[I]''', '''def kind_labeller(arm, lane, raw):
    """kinds of delivered points: g0 = the kinds of the delivered (old GT) polylines themselves, else the kind of the nearest lane GT point (its segment)"""
    if arm == "g0":
        pk = np.asarray([k == "connector" for k in raw["kind"]], int)
        return lambda P, I: pk[I]''')
rep('''    curves = pred["links"]["curves"]
    for l, (path, lg) in enumerate(zip(nt["path"], nt["long"])):
        gtp''', '''    curves = pred["links"]["curves"]
    for l, (path, lg) in enumerate(zip(nt["path"], nt["long"])):
        if l >= len(curves):                                                       # a prediction with fewer links than the table (planted degradations)
            break
        gtp''')
rep('''def frame_record(arm, f, pred=None):
    """everything one frame of `arm` contributes: core = the scorer's counts (qgraph_eval.graph_counts against the old graph GT), split (b), and for the decoded arms links (c), supp (d), out (e),
    coll, time (oracle, read-out seconds). `pred` replaces the oracle prediction (planted degradations). The lane GT is loaded here and let go again."""''',
'''def frame_record(arm, f, override=None):
    """everything one frame of `arm` contributes: core = the scorer's counts (qgraph_eval.graph_counts against the old graph GT), split (b), and for the decoded arms links (c), supp (d), out (e),
    coll, time (oracle, read-out seconds). `override` replaces the arm's graph (g0 / g1: a raw graph dict) or the oracle prediction (l2 / l1: a prediction dict with its `table`): the planted
    degradations of the self-test. The lane GT is loaded here and let go again."""''')
rep('''        if arm == "g0":
            graph = raw = gt
        elif arm == "g1":
            graph = raw = lane_polylines(lane)
        else:
            pred, graph, t_or, t_ro = decoded(lane, pred)''', '''        if arm == "g0":
            graph = raw = gt if override is None else override
        elif arm == "g1":
            graph = raw = lane_polylines(lane) if override is None else override
        else:
            pred, graph, t_or, t_ro = decoded(lane, override)''')
rep('''rec["split"] = split_counts(raw, gt, kind_labeller(arm, lane, gt))''', '''rec["split"] = split_counts(raw, gt, kind_labeller(arm, lane, raw))''')
rep('''    rec = dict(core''' if False else '''    rec = {}
    try:''', '''    rec = {}
    try:''')
open(p, "w", encoding="utf-8").write(s)

p = "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e346/e346_arm.py"
s = open(p, encoding="utf-8").read()
rep('''    if n is None and out and os.path.realpath(out).startswith(os.path.realpath(CARDS)) is False:
        pass
''', '')
rep('''    with L.cell_size(L.GRID.get(arm, QT.RES)):
        L.check_cell_size(QT.RES) if arm in L.GRID else None
        for i, f''', '''    res = L.GRID.get(arm, QT.RES)
    with L.cell_size(res):
        if arm in L.GRID:
            L.check_cell_size(res)
        for i, f''')
rep('''    print(f"E346 VAL, {', '.join(f'{a}: {c['frames']} frames' for a, c in cards.items())}\n")''', '''    sizes = ", ".join("%s: %d frames" % (a, c["frames"]) for a, c in cards.items())
    print(f"E346 VAL, {sizes}\n")''')
rep('''        import qnodes
        assert max(max(v.stop - 1 if isinstance(v, slice) else v for v in qnodes.CH.values()) + 1 for _ in (0,)) == 16
        g, _ = QE.toy_lane_gt()''', '''        import qnodes
        assert max(v.stop - 1 if isinstance(v, slice) else v for v in qnodes.CH.values()) + 1 == 16
        g, _ = QE.toy_lane_gt()''')
rep('''(f"{mod.__name__} holds a copy that hides its name", types.SimpleNamespace(**vars(stale)))''', '''(f"{mod.__name__} holds a copy that hides its name", types.SimpleNamespace(**{k: v for k, v in vars(stale).items() if not k.startswith("__")}))''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
