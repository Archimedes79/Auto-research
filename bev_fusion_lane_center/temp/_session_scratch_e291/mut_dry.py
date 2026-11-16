import re, sys
sys.path.insert(0, "temp/e291")
import mutate_e291 as M
R = M.R
for mid, f, pat, rep, test, what in M.M:
    path = f"{R}/{f}" if f.startswith("temp/e291") or f == M.JPR else f"{M.STAGE}/{f}"
    s = open(path, encoding="utf-8").read()
    ok = all(len(re.findall(p_, s)) == 1 for p_ in (pat if isinstance(pat, list) else [pat]))
    if not ok: print("BAD", mid)
print("dry run done", len(M.M))
