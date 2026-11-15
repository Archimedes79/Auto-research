p = "temp/e276/e276_build_report.py"
s = open(p, encoding="utf-8").read()
s = s.replace("the hard step stays the read-out meanwhile.", "the delivered chain still uses the hard step until you decide.")
s = s.replace('parts.append("<h2>5. Next</h2><ul><li>Memory for the long links', 'parts.append("<h2>5. Next</h2><ul><li><b>Your decision:</b> the soft step at 1.25 m (E279, KEEP by the written rule) as the delivered thinning in place of the hard read-out; it changes the baseline B (D58), so it needs a recorded decision. Both runs are two seeds, in-sample on VAL.</li><li>Memory for the long links')
open(p, "w", encoding="utf-8").write(s)
print("ok")
