import re, sys
sys.path.insert(0, "temp/e291")
import mutate as M
R = M.R
for mid, f, pat, rep, test, what in M.M:
    path = f"{M.STAGE}/{f}" if f != M.JPR else f"{R}/{M.JPR}"
    s = open(path, encoding="utf-8").read()
    n = len(re.findall(pat, s))
    print(("OK " if n == 1 else f"BAD({n}) "), mid)
