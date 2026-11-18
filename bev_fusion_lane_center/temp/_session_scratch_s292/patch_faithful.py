import io
p = "evaluation/jwire_data.py"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:90], s.count(a))
    s = s.replace(a, b)
rep('def live_records(seed, split="train", thr=0.1, nmax=0, workers=3, bs=4, device=None, pair_run=None, ends="kp"):',
    '''# E358 attempt 4 (card section 5, pre-stated before the run): the settings the two STORED dumps were made with, constants and not parameters; the check `jwire_check.py live --faithful` replays both
DUMP_B = dict(bs=4, map_mode="exact")                      # B's full dump `temp/e276/predfull_e273_p_s<seed>.pkl`: `qgraph_run_eval.py:89` (`LD.loader("val_clean", bs=4, ..., map_mode=<exact>, pixels=False, strict=True)`)
DUMP_PAIR = dict(bs=8, map_mode="error", workers=2)        # the pair dump `runs/<pair run>/ends_val.pkl`: `jends_eval.py:42` (`dump`: `DEFAULT_JUNCTION = True; LD.loader("val_clean", bs=8, workers=2, shuffle=False, map_mode="error", pixels=False)`)
FAITHFUL_MAX = 64                                          # the faithful pass keeps B's predictions of all its frames in memory: a check, not a TRAIN pass


def live_records(seed, split="train", thr=0.1, nmax=0, workers=3, bs=4, device=None, pair_run=None, ends="kp", faithful=False):''')
rep('''    assert split in SPLITS, f"split {split!r}: only {SPLITS} (TEST is never read by a pass of this module)"
    check_ends(ends, seed, pair_run)
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mb, mp = load_models(seed, resolve_pair_run(seed, pair_run), dev)
    import lane_gt_data as LD
''', '''    assert split in SPLITS, f"split {split!r}: only {SPLITS} (TEST is never read by a pass of this module)"
    check_ends(ends, seed, pair_run)
    assert not faithful or (split == "val_clean" and 0 < nmax <= FAITHFUL_MAX), f"faithful (the two-loader replay of the stored dumps' settings, E358) is a check of at most {FAITHFUL_MAX} val_clean frames, not a TRAIN pass"
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mb, mp = load_models(seed, resolve_pair_run(seed, pair_run), dev)
    if faithful:
        return _live_records_faithful(thr, nmax, workers, dev, mb, mp, ends)
    import lane_gt_data as LD
''')
rep('''def cache_path(kind, seed, nmax=0, thr=0.1, pair_run=None, ends="kp"):''', '''def _live_records_faithful(thr, nmax, workers, dev, mb, mp, ends):
    """E358 attempt 4: the live records of the first `nmax` val_clean frames with BOTH models run the way their stored dumps were made: B on a loader of its dump's settings (`DUMP_B`), the pair model on a loader of the
    pair dump's settings (`DUMP_PAIR`), the two joined by the frame's `row` (the GT ends of the record come from B's loader, as in `live_records`). The record construction after the two forwards is the same as there."""
    import torch
    import lane_gt_data as LD
    import qgraph_eval as QE
    import qgraph_readout as QR
    import qpredict as QP
    spec = QR.parse("close:1.5@0.5")
    old_ends = getattr(LD.QData, "DEFAULT_ENDS", False)
    LD.QData.DEFAULT_ENDS = True
    ds, dl = LD.loader("val_clean", bs=DUMP_B["bs"], workers=workers, shuffle=False, map_mode=DUMP_B["map_mode"], pixels=False, strict=True, ends=True)
    assert ds.missing == 0, f"{ds.missing} frames of val_clean have no stored lane GT"
    b_side = {}
    for b in dl:                                                                                           # pass 1: B (QP.predict) on its dump's loader
        for i, pr in enumerate(QP.predict(mb, b)):
            b_side[int(b["row"][i])] = (pr, dict(ent=b["ent"][i].numpy(), ent_m=b["ent_m"][i].numpy(), ext=b["ext"][i].numpy(), ext_m=b["ext_m"][i].numpy()))
        if len(b_side) >= nmax:
            break
    rows = list(b_side)[:nmax]
    assert len(rows) == nmax, f"B's loader gave {len(rows)} frames, {nmax} asked"
    LD.QData.DEFAULT_ENDS = old_ends
    LD.QData.DEFAULT_JUNCTION = True
    ds2, dl2 = LD.loader("val_clean", bs=DUMP_PAIR["bs"], workers=DUMP_PAIR["workers"], shuffle=False, map_mode=DUMP_PAIR["map_mode"], pixels=False)
    by_row = {}
    for b in dl2:                                                                                          # pass 2: the pair model on its dump's loader; a frame is kept when B's pass saw it
        with torch.no_grad():
            _, _, o = mp(b["x"].to(dev).float(), gt=None)
        for i in range(len(b["row"])):
            if int(b["row"][i]) in b_side and int(b["row"][i]) not in by_row:
                by_row[int(b["row"][i])] = frame_dict(o, i, ends)
        if all(r in by_row for r in rows):
            break
    assert all(r in by_row for r in rows), f"the pair loader did not reach the rows {[r for r in rows if r not in by_row]} of B's pass"
    out = []
    for r in rows:
        pr, g = b_side[r]
        pred = QR.apply(dict(nodes=dict(pos=pr["nodes"]["pos"], dirv=pr["nodes"]["dirv"], e=pr["nodes"]["e"], flags=pr["nodes"]["flags"], section=pr["nodes"]["section"], s=pr["nodes"].get("s")),
                             links=dict(src=pr["links"]["src"], dst=pr["links"]["dst"], p=pr["links"]["p"], long=pr["links"]["long"], a=pr["links"]["a"], b=pr["links"]["b"])), spec)
        graph, bk = QE.readout(pred, 0.5, 0.5)
        out.append(record_of(pred, graph, bk, by_row[r], g, thr, r, -1, ends))
    assert len({x["row"] for x in out}) == len(out) == nmax, "a row was visited twice or a frame is missing"
    return out


def cache_path(kind, seed, nmax=0, thr=0.1, pair_run=None, ends="kp"):''')
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
