p = "evaluation/bev_views.py"
s = open(p, encoding="utf-8").read()
def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:80])
    s = s.replace(old, new)
rep('composite(o["prob"], o["dist"]),', 'composite(o["prob"], o["dist"], observed=observed_area(g0)),', 1)
rep('composite(o0["prob"], o0["dist"]),', 'composite(o0["prob"], o0["dist"], observed=observed_area(g0)),', 2)
rep('img = composite(od["prob"], od["dist"])', 'img = composite(od["prob"], od["dist"], observed=observed_area(sample.reference[dt]))', 1)
rep('composite(od["prob"], od["dist"]),', 'composite(od["prob"], od["dist"], observed=observed_area(sample.reference[dt])),', 1)
s = s.replace("red boxes: score >= 0.3", "amber boxes: score >= 0.3").replace("(red boxes >= 0.3,", "(amber boxes >= 0.3,").replace("dashed black: GT", "dashed white: GT")
open(p, "w", encoding="utf-8").write(s)
print("ok")
