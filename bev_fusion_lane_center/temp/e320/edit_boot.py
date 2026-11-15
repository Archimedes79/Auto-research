def edit(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)

edit("temp/e320/dev/qends_eval.pysrc", [
    ('''    json.dump(out, open(os.path.join(run_dir, name), "w"), indent=1)
''', '''    json.dump(out, open(os.path.join(run_dir, name), "w"), indent=1)
    pickle.dump(dict(scenes=[f.scene for f in frames], base=rec["base"], head=rec["head"]), open(os.path.join(run_dir, name[:-5] + ".counts.pkl"), "wb"))        # the per-frame counts: `jdiag.py boot` resamples them by scene (paired interval of two read-outs, E320)
'''),
])
edit("temp/e320/dev/jdiag.pysrc", [
    ('''    wsl bash tools/env/_py14.sh jdiag.py selftest
''', '''    wsl bash tools/env/_py14.sh jdiag.py boot   e320_a_s0 ends_card_snap_nogeom_attach_thr0.1 e312_p_s0 ends_card_snap_thr0.1
                                                                                    paired scene bootstrap (1000 resamples of the VAL scenes) of the difference of two read-outs on the same frames: topo F1, junction F1, F1@1.0, lanes found (E320: a - c per seed; reads runs/<run>/<card stem>.counts.pkl that `score` writes)
    wsl bash tools/env/_py14.sh jdiag.py selftest
'''),
    ('''def maps_summary(d):''', '''def boot(ra, ca, rb, cb, n=1000, seed=0):
    """paired scene bootstrap of (read-out A - read-out B) on the same VAL frames: the pooled topo F1, junction F1, F1@1.0 and lanes found over resampled scenes (frames of a scene stay together), from the per-frame counts of the two score cards
    (`runs/<run>/<card stem>.counts.pkl`, written by `qends_eval.cmd_score`); prints the point difference, the bootstrap mean and the 95 % interval. The rule of a card stays the point estimate; this is the interval next to it."""
    import qgraph_eval as QE
    A, B = (pickle.load(open(os.path.join(ROOT, "runs", r, c + ".counts.pkl"), "rb")) for r, c in ((ra, ca), (rb, cb)))
    assert A["scenes"] == B["scenes"], "the two read-outs must be scored on the same frames"
    sc = np.asarray(A["scenes"])
    uniq = sorted(set(sc.tolist()))
    where = {s: np.where(sc == s)[0] for s in uniq}
    keys = ("topo_F1", "junction_F1", "F1@1.0", "lanes_found")
    pa, pb = QE.graph_numbers(A["head"]), QE.graph_numbers(B["head"])
    rng = np.random.RandomState(seed)
    d = {k: [] for k in keys}
    for _ in range(n):
        pick = np.concatenate([where[uniq[i]] for i in rng.randint(0, len(uniq), len(uniq))])
        na, nb = QE.graph_numbers([A["head"][i] for i in pick]), QE.graph_numbers([B["head"][i] for i in pick])
        for k in keys:
            d[k].append(na[k] - nb[k])
    print(f"A = {ra} {ca}   B = {rb} {cb}   {len(uniq)} scenes, {n} resamples")
    for k in keys:
        x = np.asarray(d[k])
        print(f"  {k:12s} A - B: point {pa[k] - pb[k]:+.4f}, bootstrap mean {x.mean():+.4f}, 95 % [{np.percentile(x, 2.5):+.4f}, {np.percentile(x, 97.5):+.4f}]")


def maps_summary(d):'''),
    ('''    elif cmd == "snaprule" and len(sys.argv) > 2:''', '''    elif cmd == "boot" and len(sys.argv) == 6:
        boot(*sys.argv[2:6])
    elif cmd == "snaprule" and len(sys.argv) > 2:'''),
])
print("boot ok")
