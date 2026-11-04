def edit(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:80])
        s = s.replace(old, new, 1)
    open(p, "w", encoding="utf-8").write(s)


edit("evaluation/bev_adapter.py", [
    ('''            TapSpec("motion_l0", "feature_map", "motion", "motion block, coarsest level: |v| smooth, bright on moving objects", "uniform, or a field along the ego motion", **kw),
            TapSpec("motion_l1", "feature_map", "motion", "motion block, middle level: the coarse field refined", "noise, or the coarse field copied", **kw),
            TapSpec("motion_l2", "feature_map", "motion", "motion block, finest level = v_hat: moving objects bright, static cells ~ 0", "bright on static cells (ego motion not removed)", **kw),''',
     '''            TapSpec("motion_l0", "feature_map", "motion", "motion block, coarsest level: |v| smooth, bright on moving objects", "uniform, or a field along the ego motion", **kw),
            TapSpec("motion_l1", "feature_map", "motion", "motion block, second level (the finest when the block has two): the coarse field refined", "noise, or the coarse field copied", **kw),
            TapSpec("motion_l2", "feature_map", "motion", "motion block, FINEST level whatever the number of levels (= v_hat): moving objects bright, static cells ~ 0",
                    "bright on static cells (ego motion not removed)", **kw),'''),
    ('''                if "hurdle_mov" in names and "mov" in res["out"][0.0]["obj"]:
                    arrays["hurdle_mov"] = res["out"][0.0]["obj"]["mov"][0].float().cpu().numpy()
                if "radar_aux" in names and getattr(self.model, "aux_rad", None) is not None:        # the head is only run in training; here it reads the branch's features
                    feat = arrays.get("radar_memory", arrays.get("radar_feat"))                      # the branch's output: the memory's when on
                    if feat is not None:
                        arrays["radar_aux"] = self.model.aux_rad(torch.from_numpy(feat)[None].to(self.device))[0].float().cpu().numpy()
''', '''                output_taps(self.model, res["out"][0.0], arrays, names, self.device)
'''),
    ('''def _first_of_pair(o):''', '''def output_taps(model, out0, arrays, names, device):
    """the taps that are not a module's output, added to `arrays` in place: `hurdle_mov` (the object head's moving logits per class of dt 0) and `radar_aux` (the radar
    auxiliary head, which only runs while training, applied here to the radar branch's output: the radar memory's when the memory is on, the radar encoder's
    otherwise; with the memory on but its tap unreadable the tap is skipped, not fed with the wrong stage)"""
    import torch
    if "hurdle_mov" in names and "mov" in out0["obj"]:
        arrays["hurdle_mov"] = out0["obj"]["mov"][0].float().cpu().numpy()
    if "radar_aux" in names and getattr(model, "aux_rad", None) is not None:
        feat = arrays.get("radar_memory") if getattr(model, "rad_temporal", None) is not None else arrays.get("radar_feat")
        if feat is not None:
            arrays["radar_aux"] = model.aux_rad(torch.from_numpy(feat)[None].to(device))[0].float().cpu().numpy()


def _first_of_pair(o):'''),
])

edit("evaluation/bev_selftest.py", [
    ('''        def forward(self, det, with_state=True):
            f = torch.ones(2, 3, 4, 4) * 5
            return (f, torch.ones(2, 3, 4, 4) * 7) if self.pair else f''', '''        def forward(self, det, with_state=True):
            f = torch.ones(2, 3, 4, 4) * 5 + torch.arange(2.0)[:, None, None, None]            # sample 1 differs from sample 0: a batch slice instead of the features is seen
            return (f, torch.ones(2, 3, 4, 4) * 7 + torch.arange(2.0)[:, None, None, None]) if self.pair else f'''),
    ('''        assert float(got["radar_feat"].mean()) == 5.0 and float(got["radar_memory"].mean()) == 10.0, (pair, {k: float(v.mean()) for k, v in got.items()})
        assert ("radar_state" in got) == pair and (not pair or float(got["radar_state"].mean()) == 7.0), (pair, sorted(got))''',
     '''        assert got["radar_feat"].shape == (3, 4, 4) and float(got["radar_feat"].mean()) == 5.0 and float(got["radar_memory"].mean()) == 10.0, (pair, {k: (v.shape, float(v.mean())) for k, v in got.items()})
        assert ("radar_state" in got) == pair and (not pair or (got["radar_state"].shape == (3, 4, 4) and float(got["radar_state"].mean()) == 7.0)), (pair, sorted(got))'''),
    ('''    taps = {t.name: t for t in A.bev_taps(''', '''    # the taps that are no module's output: the hurdle's logits and the radar aux head on the right stage (the memory's output when the memory is on)
    class Aux(nn.Module):
        def forward(self, x):
            return x[:, :2] * 10                                                    # (1, 2, X, Y): the first two channels, scaled

    class Head(nn.Module):
        def __init__(self, memory):
            super().__init__()
            self.aux_rad = Aux()
            self.rad_temporal = Memory() if memory else None
    out0 = {"obj": {"mov": torch.arange(2 * 4 * 4.0).reshape(1, 2, 4, 4)}}
    feat, mem = np.full((3, 4, 4), 1.0, np.float32), np.full((3, 4, 4), 2.0, np.float32)
    for memory, want in ((False, 10.0), (True, 20.0)):
        arrays = {"radar_feat": feat, "radar_memory": mem}
        A.output_taps(Head(memory), out0, arrays, {"hurdle_mov", "radar_aux"}, "cpu")
        assert arrays["hurdle_mov"].shape == (2, 4, 4) and arrays["hurdle_mov"][1, 0, 0] == 16.0
        assert arrays["radar_aux"].shape == (2, 4, 4) and float(arrays["radar_aux"].mean()) == want, (memory, arrays["radar_aux"].mean())
    arrays = {"radar_feat": feat}                                                   # memory on but its tap unreadable: skipped, not fed with the encoder's features
    A.output_taps(Head(True), out0, arrays, {"radar_aux"}, "cpu")
    assert "radar_aux" not in arrays
    arrays = {}
    A.output_taps(Head(False), {"obj": {}}, arrays, {"hurdle_mov", "radar_aux"}, "cpu")                 # no mov head, no radar tap: nothing added, no error
    assert arrays == {}
    taps = {t.name: t for t in A.bev_taps('''),
])

edit("evaluation/bev_suite.py", [
    ('''    if a.command[0] == "modules":                              # CPU unless --device is given: the probe must not disturb a training on the GPU
        from evaluation import bev_modules
        return bev_modules.main([a.run, "--device", a.device if "--device" in argv else "cpu", "--split", a.split] + (["--out", a.out] if a.out else []) + a.command[1:])
    if a.command[0] == "timing":
        from evaluation import bev_timing
        return bev_timing.main([a.run, "--device", a.device] + (["--out", a.out] if a.out else []) + a.command[1:])''',
     '''    if a.command[0] in ("modules", "timing"):                  # these two take --device, --split (modules), --ckpt-dir and --out; the other global options would be ignored: refused
        if a.control or (a.command[0] == "timing" and a.split != "val"):
            p.error(f"`{a.command[0]}` does not use --control" + (" or --split" if a.command[0] == "timing" else ""))
        common = (["--ckpt-dir", a.ckpt_dir] if a.ckpt_dir else []) + (["--out", a.out] if a.out else [])
        if a.command[0] == "modules":                          # CPU unless --device is given: the probe must not disturb a training on the GPU
            from evaluation import bev_modules
            explicit = any(x == "--device" or x.startswith("--device=") for x in argv)
            return bev_modules.main([a.run, "--device", a.device if explicit else "cpu", "--split", a.split] + common + a.command[1:])
        from evaluation import bev_timing
        return bev_timing.main([a.run, "--device", a.device] + common + a.command[1:])'''),
])
print("ok")
