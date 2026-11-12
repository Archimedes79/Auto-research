p = "training/grid/jends.py"
s = open(p, encoding="utf-8").read()
old = "    # the crossing penalty: P = 1 on the crossed pair set"
new = ('''    # E303: the scale head (two handle scales per pair) trains against the fitted scales of the true pairs
    m = JEnds(d=d, order=True, evidence=True, scales=True)
    out = m(jout, P, gt=gt)
    assert out["scale"].shape == (B, K, K, 2) and float(out["scale"].min()) >= 0.1 and float(out["scale"].max()) <= 0.85
    batch["pscale"] = torch.full((B, 2, 2), 0.3)
    loss, info = jends_loss(out, batch)
    loss.backward()
    assert "scale_l1" in info and all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.scale.parameters()), info
    batch.pop("pscale")
    _, info2 = jends_loss(out, batch)
    assert "scale_l1" not in info2
''' + old)
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
