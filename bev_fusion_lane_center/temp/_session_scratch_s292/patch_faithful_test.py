import io
p = "evaluation/jwire_data.py"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:90], s.count(a))
    s = s.replace(a, b)
rep("    _test_ends()\n    print(\"jwire_data selftest ok\")", "    _test_ends()\n    _test_faithful()\n    print(\"jwire_data selftest ok\")")
test = '''def _test_faithful():
    """E358 attempt 4 (`live_records(..., faithful=True)`): both models run on loaders with the settings of THEIR stored dumps (`DUMP_B`, `DUMP_PAIR`), the two joined by the frame's row although the batches differ in size, the
    GT ends of the record come from B's loader, the flags of the loader class are set for the pair loader the way the dump set them, it is a check of at most FAITHFUL_MAX val_clean frames, and the default path is untouched"""
    import types
    from types import SimpleNamespace as NS
    import torch
    rows = [10, 11, 12, 13, 14, 15]
    calls, seen = [], []

    def make(bs, coded):
        out = []
        for a in range(0, len(rows), bs):
            r = rows[a:a + bs]
            t = torch.tensor(r, dtype=torch.float32)
            out.append(dict(x=t[:, None], row=torch.tensor(r), ent=t[:, None, None].repeat(1, 1, 4), ent_m=torch.zeros(len(r), 1, dtype=torch.bool), ext=t[:, None, None].repeat(1, 1, 4), ext_m=torch.zeros(len(r), 1, dtype=torch.bool)))
        return out
    QData = NS(DEFAULT_ENDS=False, DEFAULT_JUNCTION=False)

    def loader(split, bs=0, workers=0, shuffle=None, map_mode=None, pixels=None, strict=False, ends=False):
        calls.append(dict(split=split, bs=bs, workers=workers, shuffle=shuffle, map_mode=map_mode, pixels=pixels, strict=strict, ends=ends, flags=(QData.DEFAULT_ENDS, QData.DEFAULT_JUNCTION)))
        return NS(missing=0), make(bs, True)
    fakes = dict(lane_gt_data=NS(QData=QData, loader=loader), qgraph_eval=NS(readout=lambda pred, a, b: (pred, "bk")), qgraph_readout=NS(parse=lambda s: s, apply=lambda d, spec: d["nodes"]["pos"]),
                 qpredict=NS(predict=lambda mb, b: [dict(nodes=dict(pos=float(b["x"][i, 0]), dirv=0, e=0, flags=0, section=0), links=dict(src=0, dst=0, p=0, long=0, a=0, b=0)) for i in range(len(b["row"]))]))

    def model(x, gt=None):                                                                                 # the pair model: every output is coded with the frame's row
        n = x.shape[0]
        o = dict(kp_pos=x[:, :, None].repeat(1, 4, 2), kp_dir=torch.zeros(n, 4, 2), kp_p=torch.zeros(n, 4), valid=torch.ones(n, 2, 2, dtype=torch.bool), logit=torch.zeros(n, 2, 2), allowed=torch.ones(n, 2, 2, dtype=torch.bool))
        return None, None, o
    saved = {k: sys.modules.get(k) for k in fakes}
    real_record_of, real_lm = record_of, load_models
    try:
        for k, v in fakes.items():
            sys.modules[k] = types.ModuleType(k)
            sys.modules[k].__dict__.update(vars(v))
        globals()["record_of"] = lambda pred, graph, bk, e, g, thr=0.1, row=-1, scene=-1, ends="kp": seen.append((pred, float(e["kp_pos"][0, 0]), float(g["ent"][0, 0]), row, ends)) or toy_record(row, scene)
        globals()["load_models"] = lambda seed, pair_run, dev: ("B", model)
        recs = live_records(0, "val_clean", 0.1, 5, device="cpu", pair_run="e324_s_s0", ends="att", faithful=True)
        assert [r["row"] for r in recs] == rows[:5], [r["row"] for r in recs]
        assert seen == [(float(r), float(r), float(r), r, "att") for r in rows[:5]], seen                      # B's prediction, the pair model's output and the GT ends of every record belong to the SAME row
        assert len(calls) == 2, calls
        b_call, p_call = calls
        assert (b_call["split"], b_call["bs"], b_call["map_mode"], b_call["pixels"], b_call["strict"], b_call["ends"], b_call["shuffle"]) == ("val_clean", DUMP_B["bs"], DUMP_B["map_mode"], False, True, True, False) == ("val_clean", 4, "exact", False, True, True, False), b_call
        assert (p_call["split"], p_call["bs"], p_call["workers"], p_call["map_mode"], p_call["pixels"], p_call["shuffle"]) == ("val_clean", DUMP_PAIR["bs"], DUMP_PAIR["workers"], DUMP_PAIR["map_mode"], False, False) == ("val_clean", 8, 2, "error", False, False), p_call
        assert b_call["flags"] == (True, False) and p_call["flags"] == (False, True), (b_call["flags"], p_call["flags"])      # B's loader: ends flag on; the pair loader as the dump set it: junction flag on, the ends flag back where it was
        assert DUMP_B == dict(bs=4, map_mode="exact") and DUMP_PAIR == dict(bs=8, map_mode="error", workers=2) and FAITHFUL_MAX == 64
        # the guards: only a check of val_clean frames, never a TRAIN pass, never without nmax or above FAITHFUL_MAX
        n_calls = len(calls)
        for kw in (dict(split="train", nmax=5), dict(split="val_clean", nmax=0), dict(split="val_clean", nmax=FAITHFUL_MAX + 1)):
            assert "faithful" in (refusal(live_records, 0, kw["split"], 0.1, kw["nmax"], device="cpu", pair_run="e324_s_s0", ends="att", faithful=True) or ""), kw
        assert len(calls) == n_calls, "refused before any loader was built"
        # a pair loader that never reaches a row of B's pass is an error, not a short result
        globals()["load_models"] = lambda seed, pair_run, dev: ("B", model)
        short_rows, rows[:] = rows[:], rows[:4]
        try:
            fakes_loader = sys.modules["lane_gt_data"].loader
            first = []

            def loader2(split, bs=0, **k):
                ds, b = fakes_loader(split, bs=bs, **k)
                first.append(bs)
                if len(first) == 2:
                    b = b[:0]                                                                              # the pair loader gives nothing
                return ds, b
            sys.modules["lane_gt_data"].loader = loader2
            rows[:] = short_rows
            assert "pair loader" in (refusal(live_records, 0, "val_clean", 0.1, 5, device="cpu", pair_run="e324_s_s0", ends="att", faithful=True) or ""), "missing rows on the pair side"
        finally:
            sys.modules["lane_gt_data"].loader = fakes_loader
            rows[:] = short_rows
        # the default path (faithful off) builds ONE loader, exactly as before
        calls.clear()
        live_records(0, "val_clean", 0.1, 4, device="cpu", pair_run="e324_s_s0", ends="att")
        assert len(calls) == 1 and calls[0]["bs"] == 4 and calls[0]["map_mode"] == "exact", calls
    finally:
        globals()["record_of"], globals()["load_models"] = real_record_of, real_lm
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _selftest():'''
rep("def _selftest():", test)
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
