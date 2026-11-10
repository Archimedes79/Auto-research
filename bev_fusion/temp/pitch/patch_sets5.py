p = "evaluation/evaluate.py"
s = open(p, encoding="utf-8").read()
old = '''chain of a finished run: generalisation, report page) and `--quant --curves ...` are for the run's own lists, a named set takes --quant / --qual / --videos / --time. The TEST
split is not offered.'''
new = '''chain of a finished run: generalisation, the report page) is for the run's own lists; a named set takes --quant (with --curves / --fusion / --velocity), --qual, --videos and
--time. The TEST split is not offered.'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("ok")
