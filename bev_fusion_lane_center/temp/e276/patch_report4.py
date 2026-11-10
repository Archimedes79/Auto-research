p = "temp/e276/e276_build_report.py"
s = open(p, encoding="utf-8").read()
a = s.index('("<div class=\'note\'><pre>" + html.escape(verd[-1800:])')
b = s.index('"<p><b>Arm s failed its gate</b>')
s = s[:a] + '("<p>Verdict of the written rule (primary: F1 at 1 m on the own lane and its neighbours, band 0.02): <b>" + (verdict_of("== verdict276 t against") or "pending") + "</b>.</p>") +\n             ' + s[b:]
a = s.index('("<div class=\'note\'><pre>" + html.escape(verd[-3200:])')
b = s.index("\n", a)
s = s[:a].rstrip().rstrip("+").rstrip() + ")" + s[b:] if False else s
# simpler: neutralise the section-3 block
s = s.replace('("<div class=\'note\'><pre>" + html.escape(verd[-3200:]) + "</pre></div>" if "verdict277" in verd else ""))', '"")')
open(p, "w", encoding="utf-8").write(s)
print("ok")
