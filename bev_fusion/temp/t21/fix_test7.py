p = "tests/test_stream_temporal.py"
s = open(p, encoding="utf-8").read()
a = s.index('    clip = [with_gt(mk(idx), m.grid) for idx in ([0, 1, 2], [3, 4, 5]) if kind == "recurrent" or True]')
b = s.index('    cfg = T.TrainConfig(')
s = s[:a] + '''    clip = [with_gt(mk(idx), m.grid) for idx in (([0, 1, 2], [3, 4, 5]) if kind == "recurrent" else ([1, 2], [4, 5]))]
''' + s[b:]
open(p, "w", encoding="utf-8", newline="").write(s)
