p = "evalsuite/apicli.py"
s = open(p, encoding="utf-8").read()
old = '''    return ", ".join(str(p).replace(" ", "") if p.default is not inspect.Parameter.empty else p.name
                     for k, p in inspect.signature(getattr(Evaluation, name)).parameters.items() if k != "self")'''
new = '''    return ", ".join(p.name if p.default is inspect.Parameter.empty else f"{p.name}={p.default!r}"
                     for k, p in inspect.signature(getattr(Evaluation, name)).parameters.items() if k != "self")'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("ok")
