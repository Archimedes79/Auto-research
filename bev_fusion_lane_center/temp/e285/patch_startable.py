p = "training/startable.py"
s = open(p, encoding="utf-8").read()
a = '''    for k in ("name", "steps"):
        if k not in cfg:
            problems.append(f"missing key {k!r}")
    for p in (inputs if inputs is not None else INPUTS[design]):'''
b = '''    for k in REQUIRED[design]:
        if k not in cfg:
            problems.append(f"missing key {k!r}")
    if inputs is None and design == "qgraph_head":
        n = lane_store_note(root)
        if n:
            notes.append(n)
    for p in (inputs if inputs is not None else inputs_of(cfg)):'''
assert a in s
s = s.replace(a, b)
a2 = '''    out = dict(cfg)
    out.update(SMOKE)
    out["name"] = "smoke_" + cfg["name"]'''
b2 = '''    out = dict(cfg)
    out.update(SMOKE_OF.get(cfg.get("design"), SMOKE))
    if cfg.get("design") == "qgraph_head":
        out.pop("steps", None)
    out["name"] = "smoke_" + cfg["name"]'''
assert a2 in s
s = s.replace(a2, b2)
a3 = '''    s = smoke_of(good)
    assert s["steps"] == 40'''
b3 = '''    q = dict(name="e273_p_s0", design="qgraph_head", seed=0, workers=3, node_w_evt=3.0, sparse=[0.3, 7.5])
    assert not check(q, root=tmp0, inputs=[])[0]
    for bad, frag in (({"phases": [1]}, "phases 2..7"), ({"tier_w": [1, 2]}, "three factors"), ({"node_loss": "x"}, "e273 or zone"), ({"steps": [0]}, "steps > 0"), ({"stepz": 1}, "unknown key")):
        p, _ = check(dict(q, **bad), root=tmp0, inputs=[])
        assert any(frag in x for x in p), (bad, p)
    assert "runs/e271_a_s1/model.pt" in inputs_of(dict(q, seed=1)) and "runs/e271_a_s0/model.pt" not in inputs_of(dict(q, head="x.pt"))
    qs = smoke_of(q)
    assert qs["smoke"] and "steps" not in qs and qs["name"] == "smoke_e273_p_s0" and not check(qs, root=tmp0, inputs=[])[0]
    s = smoke_of(good)
    assert s["steps"] == 40'''
assert a3 in s
s = s.replace(a3, b3)
open(p, "w", encoding="utf-8").write(s)
