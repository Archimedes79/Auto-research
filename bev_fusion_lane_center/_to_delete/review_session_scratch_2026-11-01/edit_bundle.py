p = "provenance.py"
s = open(p, encoding="utf-8").read()
a = "def bundle(code_dir, roots=BUNDLE_ROOTS, into=None, config=None):"
b = "def bundle(code_dir, roots=BUNDLE_ROOTS, into=None, config=None, exclude=BUNDLE_EXCLUDE):"
assert s.count(a) == 1
s = s.replace(a, b)
a = "    files = import_closure(code_dir, roots)\n    if not files:"
b = "    skipped = set()\n    files = import_closure(code_dir, roots, exclude, skipped)\n    if not files:"
assert s.count(a) == 1
s = s.replace(a, b)
a = '"roots": list(roots), "source":'
b = '"roots": list(roots), "excluded": list(exclude), "imports_excluded": sorted(skipped), "source":'
assert s.count(a) == 1
s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
