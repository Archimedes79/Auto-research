p = "training/grid/qpredict.py"
s = open(p, encoding="utf-8").read()
old = s[s.index("def load_model(run_dir, device=\"cuda\"):"): s.index("def _selftest(run):")]
new = '''def load_model(run_dir, device="cuda"):
    """a trained run (`model.pt`) or, while it trains, its last checkpoint (`ckpt.pt`)"""
    mp = os.path.join(run_dir, "model.pt")
    if os.path.exists(mp):
        sd = torch.load(mp, map_location="cpu")
        a, state = sd.get("args") or {}, sd["state"]
    else:
        sd = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
        a, state = {}, sd["model"]
    m = QG.QGraph(width=int(a.get("width", 48)))
    m.load_state_dict(state)
    return m.to(device).eval()


'''
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
