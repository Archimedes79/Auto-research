p = "temp/e320/t_real.py"
s = open(p, encoding="utf-8").read()
old = '''        gt = {k: b[k] for k in ("ent", "ext", "ent_m", "ext_m")}
        _, jout, out = m(b["x"].float(), gt=gt)
        loss, info = JE.jends_loss(out, b, cross_w=1.0, driven_w=0.0, geom_w=0.5)
        lm, im = JM.jmaps_loss(jout, b)
        loss = loss + 1.0 * lm
        assert torch.isfinite(loss), (n, info)
        opt.zero_grad()
        loss.backward()
        ga = sum('''
new = '''        gt = {k: b[k] for k in ("ent", "ext", "ent_m", "ext_m")}
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=bool(os.environ.get("AUTOCAST"))):                         # the trainer runs the forward AND the loss under bf16 autocast on the GPU
            _, jout, out = m(b["x"].float(), gt=gt)
            loss, info = JE.jends_loss(out, b, cross_w=1.0, driven_w=0.0, geom_w=0.5)
            lm, im = JM.jmaps_loss(jout, b)
            loss = loss + 1.0 * lm
        assert torch.isfinite(loss), (n, info)
        opt.zero_grad()
        loss.backward()
        ga = sum('''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
