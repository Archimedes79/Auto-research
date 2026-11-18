import os
p = os.path.join(os.environ["BFLC_ROOT"], "training/grid/test_jmap_train.py")
s = open(p, encoding="utf-8").read()
lines = s.split("\n")
i = [k for k, l in enumerate(lines) if "the smoke maps carry another sample" in l][0]
lines[i] = '    (T0, "the smoke maps lie on another frame\'s tokens", \'pos=(b["pos"][f, idx].numpy()\', \'pos=(b["pos"][f - 1, idx].numpy()\'),'
s = "\n".join(lines)
old = '''    # arm z: the E365 loss, the counter once per step, no v'''
new = '''    d_ = torch.cdist(bm["maps"]["pos"], bm["pos"])                                                      # every smoke map sample lies within 2.5 m of a token of ITS frame (the lane tokens, 0.5 m of noise)
    assert (d_.min(-1).values[bm["maps"]["valid"]] < 2.5).all(), f"{label}: the smoke maps are not near the tokens of their frames"
    # arm z: the E365 loss, the counter once per step, no v'''
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
