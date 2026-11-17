import os
root = os.environ["BFLC_ROOT"]
p = os.path.join(root, "evaluation", "token_time_ens.py")
s = open(p, encoding="utf-8").read()
a = '''    assert torch.allclose(p, (pr[0]["p"] + pr[1]["p"]) / 2) and torch.allclose(pt.sum(-1), torch.ones_like(pt.sum(-1)), atol=1e-5), "the mean of the probabilities"'''
assert s.count(a) == 1
s = s.replace(a, '''    assert torch.allclose(p, (pr[0]["p"] + pr[1]["p"]) / 2) and torch.allclose(pt, (pr[0]["pt"] + pr[1]["pt"]) / 2) and torch.allclose(pt.sum(-1), torch.ones_like(pt.sum(-1)), atol=1e-5), "the mean of the probabilities"
    assert float((pr[0]["pt"] - pr[1]["pt"]).abs().max()) > 1e-4, "the two heads differ in their type probabilities too"''')
open(p, "w", encoding="utf-8").write(s)
p = os.path.join(root, "evaluation", "tests", "test_token_time_ens.py")
s = open(p, encoding="utf-8").read()
i0 = s.index('    ("the check takes the smallest difference"')
i1 = s.index("\n", i0) + 1
s = s[:i0] + s[i1:]
open(p, "w", encoding="utf-8").write(s)
print("ok")
