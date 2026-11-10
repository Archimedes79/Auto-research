p = "training/grid/test_qtargets.py"
s = open(p, encoding="utf-8").read()
s = s.replace("""    import glob
    from autogt import lane_gt
    d = lane_gt.store_dir() if hasattr(lane_gt, "store_dir") else None
    fs = sorted(glob.glob(os.path.join(d, "*.pkl.gz"))) if d else []
    rng = np.random.RandomState(seed)
    sel = rng.choice(len(fs), min(n, len(fs)), replace=False)
    return [lane_gt.load(os.path.basename(fs[i])[: -len(".pkl.gz")]) for i in sorted(sel)]""", """    from autogt import lane_gt
    toks = sorted(lane_gt.held(os.path.join(lane_gt.DATA, lane_gt.signature())))
    rng = np.random.RandomState(seed)
    sel = rng.choice(len(toks), min(n, len(toks)), replace=False)
    return [lane_gt.load(toks[i]) for i in sorted(sel)]""")
open(p, "w", encoding="utf-8").write(s)
