p = "temp/e320/t_real.py"
s = open(p, encoding="utf-8").read()
old = 'elif mode == "flops":'
new = '''elif mode == "regress_readout":
    # the read-out modules, ORIGINAL (temp/e320/orig = the committed 7386e0e1) against DEV, on the 8-frame smoke dump: cmd_score for the old read-outs (keypoint ends: proximity / snap / snap-nogeom / snap-all / chord) and jmaneuver for the old variants must give the same numbers
    from importlib.machinery import SourceFileLoader
    from types import SimpleNamespace
    import importlib.util
    import json
    import qgraph_eval as QGE
    orig_vf = QGE.val_frames
    QGE.val_frames = lambda: orig_vf()[:8]
    d = os.path.join(ROOT, "runs", "e320_smoke_cpu_s0")
    f8 = os.path.join(HERE, "full8.pkl")
    if not os.path.exists(f8):
        pickle.dump(pickle.load(open(os.path.join(ROOT, "temp", "e276", "predfull_e273_p_s0.pkl"), "rb"))[:8], open(f8, "wb"))

    def modsrc(folder, name):
        ld = SourceFileLoader(name, os.path.join(HERE, folder, name + ".pysrc"))
        mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, ld))
        sys.modules[name] = mod
        ld.exec_module(mod)
        return mod
    base = dict(run="e320_smoke_cpu_s0", seed=0, thr=0.1, full=f8, base=None, skip_base_check=True, chord=False, snap=False)
    sets = (dict(), dict(snap=True), dict(snap=True, snap_nogeom=True), dict(snap=True, snap_all=True), dict(chord=True), dict(snap=True, chord=True, snap_guard=True))
    res = {}
    for folder in ("orig", "dev"):
        qe = modsrc(folder, "qends_eval")
        for i, kw in enumerate(sets):
            qe.cmd_score(SimpleNamespace(**dict(base, **kw)))
            name = "ends_card_" + ("chord_" if kw.get("chord") else "") + ("snap" + "".join("_" + k[5:] for k in ("snap_all", "snap_guard", "snap_dedup", "snap_nogeom", "snap_both") if kw.get(k)) + "_" if kw.get("snap") else "") + "thr0.1.json"
            c = json.load(open(os.path.join(d, name)))
            res[(folder, i)] = (c["numbers"], c["difference"], c["curves"], c["wired_both_ends"])
        jm = modsrc(folder, "jmaneuver")
        jm.score(SimpleNamespace(run="e320_smoke_cpu_s0", seed=0, variants=["plain", "p0.5", "p0.1", "p0.1s", "oracle", "oracle+fit"], max=8, rows=False, chord=False))
        res[(folder, "man")] = json.load(open(os.path.join(d, "maneuver_val.json")))["variants"]
    for i in range(len(sets)):
        assert res[("orig", i)] == res[("dev", i)], (i, res[("orig", i)], res[("dev", i)])
    assert res[("orig", "man")] == res[("dev", "man")], (res[("orig", "man")], res[("dev", "man")])
    print("regress_readout ok: cmd_score (6 read-outs) and jmaneuver (6 variants) identical between the committed and the dev modules")
elif mode == "flops":'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
