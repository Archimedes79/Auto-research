def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8", newline="\n").write(s)
patch("bev_fusion/tests/test_centreline_eval.py", [
    ('''       "no panel for a model without the head (no `dir`) or a GT without the layer")


def meter_checks(ck):''', '''       "no panel for a model without the head (no `dir`) or a GT without the layer")
    outputs_of_checks(ck)


def outputs_of_checks(ck):
    """`bev_adapter.outputs_of`, the one place that builds a Subject's outputs (BEVSubject.predict and bev_free.FreeSubject call it): keys, the optional parts, the 1 m head"""
    ns = SimpleNamespace(decode=lambda obj, grid, groups, k, threshold: [["det"]], NUSC_GROUPS=L.NUSC_GROUPS)
    model = SimpleNamespace(grid=GRID, spec=SPEC, spec_1m=None, boxes=lambda res, dt, k, threshold, rescore: [["refined"]])
    bare = {"sem": torch.zeros(1, len(SPEC), 8, 6), "obj": {"hm": torch.zeros(1, 1, 8, 6)}}
    keys = {"prob", "dist", "mass", "dets", "detr", "vhat", "objvel"}
    o = A.outputs_of(model, {"out": {0.0: bare}}, ns, False)[0.0]
    ck(set(o) == keys and o["dets"] == ["det"] and o["detr"] is None and o["mass"] is None and o["vhat"] is None and o["objvel"] is None,
       f"outputs_of: a bare result gives the keys {sorted(keys)}, no refined boxes, no mass / velocity ({sorted(o)})")
    ck(A.outputs_of(model, {"out": {0.0: bare}}, ns, True)[0.0]["detr"] == ["refined"], "outputs_of: refine gives the refined boxes")
    full = {**bare, "mass": torch.ones(1, 1, 8, 6), "v_hat": torch.zeros(1, 2, 8, 6), "obj": {"hm": torch.zeros(1, 1, 8, 6), "vel": torch.zeros(1, 2, 8, 6)}}
    o = A.outputs_of(model, {"out": {0.0: full, 0.5: full}}, ns, False)
    ck(sorted(o) == [0.0, 0.5] and o[0.0]["mass"].shape == (8, 6) and o[0.0]["vhat"].shape == (2, 8, 6) and o[0.0]["objvel"].shape == (2, 8, 6) and "dir" not in o[0.0],
       "outputs_of: every dt, the mass and the velocity maps when the result has them, no direction without the 1 m head")
    model.spec_1m = SPEC_1M
    o = A.outputs_of(model, {"out": {0.0: {**full, "sem_1m": p_hand()["sem_1m"]}}}, ns, False)[0.0]
    ck("centreline" in o["prob"] and o["dir"]["centreline"].shape == (2, 8, 6), "outputs_of: a model with the 1 m head adds its layer to `prob` and the direction under `dir`")


def meter_checks(ck):'''),
])
print("ok")
