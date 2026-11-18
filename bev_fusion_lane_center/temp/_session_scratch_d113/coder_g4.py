p = "training/grid/test_jpast.py"
s = open(p, encoding="utf-8").read()
i0 = s.index('    ("the meter counts calls without a past"')
i1 = s.index("]\n\n\ndef one(args):")
new = '''    ("the meter counts calls without a past", "        if not past:\n            return\n        for k, v in", "        if False:\n            return\n        for k, v in"),
    ("the meter does not clear", "s, self.acc = self.acc, dict.fromkeys(SUMS, 0)", "s = self.acc"),
    ("the meter ignores the donor", "(out - x).detach(), past, self.donor)", "(out - x).detach(), past)"),
    ("u is taken from the none-key sums", 'math.sqrt(s["d2_other"] / max(s["x2_other"], 1e-30))', 'math.sqrt(s["d2_none"] / max(s["x2"], 1e-30))'),
    ("u_none is taken from the other-sample sums", 'u_none=math.sqrt(s["d2_none"] / max(s["x2"], 1e-30))', 'u_none=math.sqrt(s["d2_other"] / max(s["x2_other"], 1e-30))'),
    ("u: the reference is the none key only", "d2_other=sq(self.delta(x, pos, dirv, valid, other))", "d2_other=sq(self.delta(x, pos, dirv, valid, None))"),
    ("u: the reference is the past itself", "d2_other=sq(self.delta(x, pos, dirv, valid, other))", "d2_other=sq(self.delta(x, pos, dirv, valid, past))"),
    ("u: the roll is missing", "v.roll(shift, 0) if torch.is_tensor(v)", "v if torch.is_tensor(v)"),
    ("u: the roll is by 0", "def roll_past(past, shift=1):", "def roll_past(past, shift=0):"),
    ("u: e_cur is rolled with the past", 'if torch.is_tensor(v) and k != "e_cur" else v', "if torch.is_tensor(v) else v"),
    ("u: a batch of one sample compares with itself", "if past and x.shape[0] >= 2 else None", "if past and x.shape[0] >= 1 else None"),
    ("u: the donor is ignored", "other = donor if donor is not None else (", "other = ("),
    ("u: the other-sample count is not kept", "x2_other=x2, n_other=n", "x2_other=x2, n_other=0"),
    ("u_none: the none-only update is the update with the past", "d2_none=sq(self.delta(x, pos, dirv, valid, None))", "d2_none=sq(self.delta(x, pos, dirv, valid, past))"),
    ("u: padding tokens count in x", "x2, n = float(((x * m) ** 2).sum()), int(valid.sum())", "x2, n = float((x ** 2).sum()), int(valid.sum())"),
    ("u: padding tokens count in the difference", "m = valid.to(x.dtype)[..., None]\n        sq = ", "m = torch.ones_like(x[..., :1])\n        sq = "),
'''
s = s[:i0] + new + s[i1:]
s = s.replace("the gradient of every parameter, K = 512 memory), then", "the gradient of every parameter, a block that ignores the past content, u and u_none, K = 512 memory), then")
open(p, "w", encoding="utf-8").write(s)
print("ok")
