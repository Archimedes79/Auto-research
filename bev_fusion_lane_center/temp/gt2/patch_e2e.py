def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:80])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, count))


p = "training/train/jends_train.py"
# the header text
sub(p, "evidence along the Hermite curve, loss = BCE + dual softmax + the soft crossing penalty). Output",
    "evidence along the Hermite curve, loss = BCE + dual softmax + the soft crossing penalty). `--e2e` (E305, D82 \"bleibe differenzierbar\"): the maps head is trained TOGETHER with the pair head, the pair loss reaches the heat (soft-argmax\nof the keypoints, heading, features, evidence samples) and the maps loss keeps the maps meaningful (`--maps-w`); the pyramid of the baseline stays frozen. Output")
sub(p, "    def __init__(self, maps_args, order=True, evidence=True, d=64, scales=False):\n        super().__init__()\n",
    "    def __init__(self, maps_args, order=True, evidence=True, d=64, scales=False, e2e=False):\n        super().__init__()\n        self.e2e = e2e\n")
sub(p, "        with torch.no_grad():\n            P, out = self.jmodel(x, joint=False)\n        return P, out, self.je(out, P, gt=gt)",
    "        if self.e2e:                                                                                     # the maps head (jmodel.jm) gets the gradient of the pair loss\n            P, out = self.jmodel(x, joint=False)\n        else:\n            with torch.no_grad():\n                P, out = self.jmodel(x, joint=False)\n        return P, out, self.je(out, P, gt=gt)")
sub(p, '    ap.add_argument("--steps", type=int, default=3000)\n', '    ap.add_argument("--e2e", action="store_true", help="E305: train the maps head together with the pair head (the pair loss reaches the heat)")\n    ap.add_argument("--maps-w", type=float, default=1.0, help="--e2e: weight of the junction maps loss next to the pair loss")\n    ap.add_argument("--steps", type=int, default=3000)\n')
sub(p, "model = JEndModel(ck[\"args\"], order=not a.no_order, evidence=not a.no_evidence, scales=a.scales)", "model = JEndModel(ck[\"args\"], order=not a.no_order, evidence=not a.no_evidence, scales=a.scales, e2e=a.e2e)")
sub(p, "    ps = list(model.je.parameters())\n    for p in model.parameters():\n        p.requires_grad_(False)\n",
    "    ps = list(model.je.parameters()) + (list(model.jmodel.jm.parameters()) if a.e2e else [])\n    for p in model.parameters():\n        p.requires_grad_(False)\n")
sub(p, 'run.log(f"jends order={not a.no_order} evidence={not a.no_evidence}: {steps} steps,', 'run.log(f"jends order={not a.no_order} evidence={not a.no_evidence} e2e={a.e2e}: {steps} steps,')
sub(p, "        _, _, out = m(x, gt=gt)\n        loss, info = JE.jends_loss(out, b, cross_w=a.cross_w, driven_w=a.driven_w)\n        return loss, info",
    "        _, jout, out = m(x, gt=gt)\n        loss, info = JE.jends_loss(out, b, cross_w=a.cross_w, driven_w=a.driven_w)\n        if a.e2e:\n            lm, im = JM.jmaps_loss(jout, b)\n            loss = loss + a.maps_w * lm\n            info.update({\"j_\" + k: v for k, v in im.items()})\n        return loss, info")
sub(p, "config=dict(design=\"jends\", order=not a.no_order, evidence=not a.no_evidence, maps=mp))\n    torch.save(dict(state=model.state_dict(), args=dict(vars(a), maps_args=ck[\"args\"])), run.path(\"model.pt\"))",
    "config=dict(design=\"jends\", order=not a.no_order, evidence=not a.no_evidence, e2e=a.e2e, maps=mp))\n    torch.save(dict(state=model.state_dict(), args=dict(vars(a), maps_args=ck[\"args\"])), run.path(\"model.pt\"))\n    if a.e2e:                                                                                          # the maps head as a plain maps run, so `jmaps_eval.py score --run <name>_maps` reads it\n        os.makedirs(os.path.join(ROOT, \"runs\", a.name + \"_maps\"), exist_ok=True)\n        torch.save(dict(state=model.jmodel.state_dict(), args=dict(ck[\"args\"], e2e_from=a.name)), os.path.join(ROOT, \"runs\", a.name + \"_maps\", \"model.pt\"))")

# the self-test: the gradient of the pair loss reaches the maps head in e2e and the frozen pyramid never
p = "evaluation/jends_eval.py"
sub(p, '    assert oe["logit"].shape == (1, 32, 32) and out["j1"].shape == (1, 8, 100, 100)\n    print("jends_eval selftest ok")',
    '''    assert oe["logit"].shape == (1, 32, 32) and out["j1"].shape == (1, 8, 100, 100)
    # E305: with e2e the pair logits reach the maps head (and never the frozen pyramid); without it the maps head gets nothing
    for e2e in (False, True):
        m = JT.JEndModel(dict(d=64, ctx="none"), order=True, evidence=True, e2e=e2e)
        m.zero_grad()
        P, out, oe = m(torch.randn(1, 39, 200, 200), gt=dict(ent=torch.zeros(1, 16, 4), ext=torch.zeros(1, 16, 4), ent_m=torch.zeros(1, 16, dtype=torch.bool), ext_m=torch.zeros(1, 16, dtype=torch.bool)))
        (oe["logit"].sum() + oe["kp_pos"].sum()).backward()
        gm = sum(float(p.grad.abs().sum()) for p in m.jmodel.jm.parameters() if p.grad is not None)
        gp = sum(float(p.grad.abs().sum()) for p in m.jmodel.pyr.parameters() if p.grad is not None)
        assert gp == 0.0, "the frozen pyramid must not get a gradient"
        assert (gm > 0) == e2e, (e2e, gm)
    print("jends_eval selftest ok")''')
print("ok")
