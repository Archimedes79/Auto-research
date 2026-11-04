p = "evaluation/bev_modules.py"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''    snap = rec.copy()
    rec.clear()
    assert not rec.store and "out" in snap["lift"]
    rec.close()
    assert m.trunk.features == plain, "the trunk is as before"''', '''    snap = rec.copy()
    rec.clear()
    assert not rec.store and "out" in snap["lift"]
    rec.close()
    assert m.trunk.features == plain and "features" not in m.trunk.__dict__, "the trunk is as before (no instance attribute left behind)"
    assert chain_modules(m) == ["lift", "radar", "trunk"] and chain_modules(nn.Sequential(nn.Identity())) == ["0"], "the chain is the model's own children"
    m.lift2 = nn.Identity()                                                      # a dual camera's second lift is probed without a list that has to know it
    rec2 = Recorder(m)
    m(torch.ones(1, 2, 3, 3))
    m.lift2(torch.ones(1))
    assert "lift2" in rec2.store
    rec2.close()
    kept = torch.ones(2)
    out_of_place = Recorder(Toy())
    out_of_place.close()
    assert stat(torch.ones(1))["std"] == 0.0, "a single element has no spread (not nan)"
    t_cpu = torch.ones(1, 2, 2, 2)
    stored = tensors_of(t_cpu)["out"]
    t_cpu.mul_(5)
    assert float(stored.mean()) == 1.0, "a later in-place operation does not change the record"
    assert changed_nothing({"a": {"x": 0.0, "y": None}}) and not changed_nothing({"a": {"x": 0.0}, "OUT": {"sem": 0.2}})''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
