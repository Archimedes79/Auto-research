import os
p = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "README.md")
s = open(p, encoding="utf-8").read()
old = "The module docstring has the dress-rehearsal command lines for VAL. "
new = ("`all` prints the digest of the imported lane code (the static import closure of the module over the project's search path, about 70 files) at its start and, with `--expect-code-digest` (+ `--expect-code-card`), refuses before any dump when it is not the code of the rehearsal; "
       "`atomic` writes only under `temp/e343/`, `runs/eval/e324s_val|test` and `runs/eval/e312_val` (an allow-list next to the deny-list of E321's folders). The module docstring has the dress-rehearsal command lines for VAL and the TEST line. ")
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("ok")
