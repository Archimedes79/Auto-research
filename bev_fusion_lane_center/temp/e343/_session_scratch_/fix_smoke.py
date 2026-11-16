import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp", "e343", "run_rehearsal_smoke_r4.sh")
s = open(p, encoding="utf-8").read()
a = s.index('echo "=== TEST line refused outside the launcher"')
b = s.index('echo "=== done"')
open(p, "w", encoding="utf-8").write(s[:a] + s[b:])
