import io
p = "temp/_session_scratch_d113/patch_critic.py"
s = io.open(p, encoding="utf-8").read()
a = '''"refined node token of the node head: the input of B's link head, which adds its own context blocks on top), f2 (float16, 144"'''
b = '''"node token of the node head: the input of B's link head, which adds its own context blocks on top), f2 (float16, 144"'''
assert s.count(a) == 1
io.open(p, "w", encoding="utf-8").write(s.replace(a, b))
