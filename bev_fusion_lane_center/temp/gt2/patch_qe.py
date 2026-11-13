p = "evaluation/qends_eval.py"
s = open(p, encoding="utf-8").read()
old = "    mine = torch.load(os.path.join(run_dir, \"model.pt\"), map_location=\"cpu\")[\"state\"]\n    base = torch.load(base_pt, map_location=\"cpu\")[\"state\"]\n    shared = [k for k in base if not k.startswith(\"ends.\")]\n    bad = [k for k in shared if k not in mine or not torch.equal(mine[k], base[k])]\n    assert not bad,"
new = "    mine = torch.load(os.path.join(run_dir, \"model.pt\"), map_location=\"cpu\")[\"state\"]\n    base = torch.load(base_pt, map_location=\"cpu\")[\"state\"]\n    shared = [k for k in base if not k.startswith(\"ends.\")]\n    bad = [] if getattr(a, \"skip_base_check\", False) else [k for k in shared if k not in mine or not torch.equal(mine[k], base[k])]      # E287: a pair head on the junction maps run is not a QGraph checkpoint (jends_eval.py says so)\n    assert not bad,"
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
