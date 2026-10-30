p = "temp/calib/prior_detector.py"
s = open(p, encoding="utf-8").read()
a = s.index('mean_size = {')
b = s.index('dets = decode(')
s = s[:a] + '''size = torch.cat([torch.full((1, 1, X, Y), 4.6), torch.full((1, 1, X, Y), 1.9)], 1)       # one size map: irrelevant for AP (centre distance)
out = {"hm": hm, "off": torch.zeros(1, 2, X, Y), "vel": torch.zeros(1, 2, X, Y), "size": size,
       "rot": torch.cat([torch.zeros(1, 1, X, Y), torch.ones(1, 1, X, Y)], 1)}
''' + s[b:]
s = s.replace('grid = type("G", (), {})()\n', '')
open(p, "w", encoding="utf-8", newline="\n").write(s)
