p = "temp/t16/soa_maps_v8.py"
s = open(p, encoding="utf-8").read()
old = '''                    w(f"| {layer} | {tol} | 0 | - | - | - | - |")'''
new = '''                    w(f"| {layer} | {tol} | 0 | - | - | - | - |" + (" - |" if rec and not best else ""))'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("ok")
