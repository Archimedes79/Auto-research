s = open("mutate.py", encoding="utf-8").read()
old = ''''use_counts = bool(setup["arms"]) and counts_reproduced(done)', 'use_counts = bool(setup["arms"])'),'''
new = ''''use_counts = counts_reproduced(done)', 'use_counts = True'),'''
assert old in s
open("mutate.py", "w", encoding="utf-8").write(s.replace(old, new))
