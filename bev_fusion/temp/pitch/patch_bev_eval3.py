p = "evaluation/bev_eval.py"
s = open(p, encoding="utf-8").read()
old = '''ran (it needs the ground truth of that moment): it is kept as it was, whatever GT a later scoring uses.
"""'''
new = '''ran (it needs the ground truth of that moment): it is kept as it was, whatever GT a later scoring uses.
Limits: the identity of a network reads its last checkpoint, so predictions of a run whose checkpoints were archived or slimmed (keep3) are found under the identity of the file that was there, and a
changed file is another network; a network is addressed by the name of its run folder (`runs/x` and `archive/x` would share the id `x@...`).
"""'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("ok")
