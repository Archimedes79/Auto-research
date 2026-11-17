p = "training/grid/jwire_head.py"
s = open(p, encoding="utf-8").read()
a = '    ck = torch.load(os.path.join(root, "runs", run, name), map_location="cpu")\n'
assert s.count(a) == 1
open(p + ".new_s292", "w", encoding="utf-8", newline="\n").write(s.replace(a, '    ck = torch.load(os.path.join(root, "runs", run, name or "model.pt"), map_location="cpu")\n'))
