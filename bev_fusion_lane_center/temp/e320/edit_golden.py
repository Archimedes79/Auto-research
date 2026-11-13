p = "temp/e320/dev/jends.pysrc"
s = open(p, encoding="utf-8").read()
block = '''    # flag-off identity that stays in the repo (critic 2026-11-12): the E312 pair head (the `je.*` weights of runs/e312_p_s0, when the run is there) on a seeded synthetic input gives the numbers pinned from the committed code of 7386e0e1 (E317 / E318 / E320 off); no attach / unc keys
    import os
    ckp = os.path.join(os.environ.get("BFLC_ROOT", ""), "runs", "e312_p_s0", "model.pt")
    if os.environ.get("BFLC_ROOT") and os.path.exists(ckp):
        st_ = {k_[3:]: v_ for k_, v_ in torch.load(ckp, map_location="cpu")["state"].items() if k_.startswith("je.")}
        mg = JEnds(d=64, order=True, evidence=True, scales=True)
        mg.load_state_dict(st_)
        mg.eval()
        gg = torch.Generator().manual_seed(7)
        jg_ = dict(j1=torch.randn(2, 8, 100, 100, generator=gg) * 0.5 - 2.5, j05=torch.randn(2, 2, 200, 200, generator=gg) - 2.0, f1=torch.randn(2, 64, 100, 100, generator=gg))
        pg_ = dict(centre1=torch.randn(2, 12, 100, 100, generator=gg))
        with torch.no_grad():
            og = mg(jg_, pg_, gt=None)
        for k_, v_ in dict(logit=-25719.9688, kp_pos=-645.8732, scale=2228.1379, kp_p=34.751).items():
            assert abs(float(og[k_].sum()) - v_) <= 1e-3 * max(1.0, abs(v_)), (k_, float(og[k_].sum()), v_)
        assert int(og["allowed"].sum()) == 1003 and int(og["valid"].sum()) == 128 and not any(k_.startswith(("att_", "kp_logsig")) for k_ in og), "the flag-off model must reproduce the E312 head"
'''
old = '''    print("jends selftest ok; params"'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, block + old))
