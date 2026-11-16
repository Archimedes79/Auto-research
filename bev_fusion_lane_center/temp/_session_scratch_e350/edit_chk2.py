p = "evaluation/jwire_check.py"
s = open(p, encoding="utf-8").read()
old = '''== ["the stored card is not the rule's (wire 'head')"]   #'''
new = '''== ["the stored card is not the rule's (wire head)"]   #'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
