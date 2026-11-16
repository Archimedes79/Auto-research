import os
P = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "jtest_eval.py")
s = open(P, encoding="utf-8").read()
a = s.index("    dirs, memo, listing = code_dirs(), {}, {}")
b = s.index("    found, todo = set(), [os.path.realpath(entry or os.path.abspath(__file__))]")
new = '''    dirs, memo, listing = code_dirs(), {}, {}

    def ls(d):                                                                                         # cached folder listings: a stat or a listdir per candidate is slow on /mnt/c
        if d not in listing:
            try:
                listing[d] = set(os.listdir(d))
            except OSError:
                listing[d] = set()
        return listing[d]

    def isfile(c):
        return os.path.basename(c) in ls(os.path.dirname(c))

    def lookup(name, roots):
        parts, out = name.split("."), []
        for d in roots:
            if parts[0] not in ls(d) and parts[0] + ".py" not in ls(d):                               # the first part is not in this folder: nothing below it to look at
                continue
            base = os.path.join(d, *parts)
            for c in (base + ".py", os.path.join(base, "__init__.py")):
                if isfile(c):
                    out += [c] + [p for p in (os.path.join(d, *parts[:k], "__init__.py") for k in range(1, len(parts))) if isfile(p)]
        return out
'''
s = s[:a] + new + s[b:]
open(P, "w", encoding="utf-8").write(s)
