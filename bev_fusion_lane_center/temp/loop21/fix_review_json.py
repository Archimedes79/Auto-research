import re, json
p = "reports/source/weekly/s21_review.json"
s = open(p, encoding="utf-8").read()
s = re.sub(r'(NEG|TIE|INFO|OK)\("([^"]*)"\)', lambda m: '"%s|%s"' % (m.group(1).lower(), m.group(2)), s)
json.loads(s)
open(p, "w", encoding="utf-8").write(s)
p = "reports/build/reports/build_status21_report.py"
b = open(p, encoding="utf-8").read()
a = 'd.table([[a, b, c] for a, b, c in REV["rows"]]'
assert b.count(a) == 1
b = b.replace(a, 'TAG = dict(ok=OK, neg=NEG, tie=TIE, info=INFO)\nd.table([[a, b, TAG[c.split("|")[0]](c.split("|", 1)[1])] for a, b, c in REV["rows"]]')
open(p, "w", encoding="utf-8").write(b)
print("ok")
