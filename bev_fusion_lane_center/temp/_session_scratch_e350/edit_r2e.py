p = "evaluation/jwire_data.py"
s = open(p, encoding="utf-8").read()
old = '''        def refused(f, *a):
            try:
                f(*a)
            except AssertionError:
                return True
            return False'''
new = '''        def refused(f, *a, **k):
            try:
                f(*a, **k)
            except AssertionError as e:
                return str(e) or True
            return False'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
