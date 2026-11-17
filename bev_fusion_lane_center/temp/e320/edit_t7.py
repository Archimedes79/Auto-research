p = "temp/e320/t_real.py"
s = open(p, encoding="utf-8").read()
old = 'elif mode == "flops":'
new = '''elif mode == "readouts_new":
    # the E320 read-outs on the 8-frame smoke dump (dev modules): primary (attached + one nearest piece per end, geometry kept), proximity, tight 1.5 / 1.0; the file names, the extra statistics, jmaneuver p0.1a / an / ar1.5 / ar1.0
    from importlib.machinery import SourceFileLoader
    from types import SimpleNamespace
    import importlib.util
    import json
    import qgraph_eval as QGE
    orig_vf = QGE.val_frames
    f8 = os.path.join(HERE, "full8.pkl")

    def modsrc(name):
        ld = SourceFileLoader(name, os.path.join(HERE, "dev", name + ".pysrc"))
        mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, ld))
        sys.modules[name] = mod
        ld.exec_module(mod)
        return mod
    qe, jm = modsrc("qends_eval"), modsrc("jmaneuver")
    base = dict(run="e320_smoke_cpu_s0", seed=0, thr=0.1, full=f8, base=None, skip_base_check=True, chord=False, snap=False)
    d = os.path.join(ROOT, "runs", "e320_smoke_cpu_s0")
    QGE.val_frames = lambda: orig_vf()[:8]
    for kw in (dict(attach=True), dict(attach=True, snap=True, snap_nogeom=True), dict(attach=True, attach_r=1.5), dict(attach=True, attach_r=1.0), dict()):
        qe.cmd_score(SimpleNamespace(**dict(base, **kw)))
    print(sorted(x for x in os.listdir(d) if x.startswith("ends_card")))
    for nme in ("ends_card_attach_thr0.1.json", "ends_card_snap_nogeom_attach_thr0.1.json", "ends_card_attach_r1.5_thr0.1.json", "ends_card_attach_r1.0_thr0.1.json", "ends_card_thr0.1.json"):
        c = json.load(open(os.path.join(d, nme)))
        print(nme, "topo", round(c["numbers"]["head"]["topo_F1"], 4), "curves", c["curves"], "wired", c["wired_both_ends"], "dropped", c["long_links_dropped"], "within 1 m", {k: round(v, 3) if isinstance(v, float) else v for k, v in c["ends_within_1m_of_a_delivered_piece_end"].items()})
    QGE.val_frames = orig_vf
    jm.score(SimpleNamespace(run="e320_smoke_cpu_s0", seed=0, variants=["plain", "p0.1", "p0.1s", "p0.1a", "p0.1an", "p0.1ar1.5", "p0.1ar1.0"], max=8, rows=False, chord=False))
elif mode == "flops":'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
