p = "training/train/qgraph_train.py"
s = open(p, encoding="utf-8").read()
s = s.replace("""    if a.resume_from:
        model.load_state_dict(torch.load(a.resume_from, map_location="cpu")["state"])""", """    if a.resume_from:                                                                       # the tensors of an earlier phase model whose shapes still fit (a changed stage keeps its fresh start)
        sd, own = torch.load(a.resume_from, map_location="cpu")["state"], model.state_dict()
        fit = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(fit, strict=False)
        print(f"resume-from {a.resume_from}: {len(fit)} of {len(own)} tensors loaded; not loaded: {sorted(set(own) - set(fit))[:6]}...", flush=True)""")
open(p, "w", encoding="utf-8").write(s)
