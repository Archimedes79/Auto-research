p = "evaluation/bev_errors.py"
s = open(p, encoding="utf-8").read()
old = '''== [("vehicle", "tp"), ("vehicle", "fn")] and bad[0][2] == 3 and bad[0][3] == 4, bad              # one more tp on the card: tp and fn (n_gt - n_tp) differ'''
new = '''== [("vehicle", "tp"), ("vehicle", "fn"), ("vehicle", "fp")] and bad[0][2] == 3 and bad[0][3] == 4, bad    # one more tp on the card: tp, n_gt - n_tp and n_det - n_tp all differ'''
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, new))
