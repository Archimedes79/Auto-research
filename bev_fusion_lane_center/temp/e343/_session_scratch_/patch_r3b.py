import os
P = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "jtest_eval.py")
s = open(P, encoding="utf-8").read()
old = '''    dirs, memo = code_dirs(), {}

    def lookup(name, roots):
        parts, out = name.split("."), []
        for d in roots:
            base = os.path.join(d, *parts)
            for c in (base + ".py", os.path.join(base, "__init__.py")):
                if os.path.isfile(c):
                    out += [c] + [p for p in (os.path.join(d, *parts[:k], "__init__.py") for k in range(1, len(parts))) if os.path.isfile(p)]
        return out'''
new = '''    dirs, memo, listing = code_dirs(), {}, {}

    def isfile(c):                                                                                     # from cached folder listings: a stat per candidate is slow on /mnt/c
        d = os.path.dirname(c)
        if d not in listing:
            try:
                listing[d] = set(os.listdir(d))
            except OSError:
                listing[d] = set()
        return os.path.basename(c) in listing[d]

    def lookup(name, roots):
        parts, out = name.split("."), []
        for d in roots:
            base = os.path.join(d, *parts)
            for c in (base + ".py", os.path.join(base, "__init__.py")):
                if isfile(c):
                    out += [c] + [p for p in (os.path.join(d, *parts[:k], "__init__.py") for k in range(1, len(parts))) if isfile(p)]
        return out'''
assert s.count(old) == 1
open(P, "w", encoding="utf-8").write(s.replace(old, new))
