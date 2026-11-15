p = "temp/_session_scratch_s292/new3/jwire_head.py"
s = open(p, encoding="utf-8").read()
def sub(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
sub("    assert torch.equal(f1[:, :2], f3[:, :2]) and torch.equal(f1, f3) or True\n", "    assert torch.equal(f1, f3) and torch.equal(f1, f2), \"a feature reads the GT\"\n")
sub("jit = lambda p, t: (p + np.array([0.0, -2.0]), t)", "jit = lambda p, t: (p + np.array([0.0, 2.0]), t)")
sub("        return logits.sum() * 0.0\n", "        return torch.nan_to_num(logits, neginf=0.0).sum() * 0.0\n")
sub("lb = (ft[:, :3, 5] > 0).float().argmax(-1)                                                          # the label = the place with the largest feature 5 among the first 3 (a rule that is not 'nearest')",
    "lb = ft[:, :3, 5].argmax(-1)                                                                        # the label = the place with the largest feature 5 among the first 3 (a rule that is not 'nearest')")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
