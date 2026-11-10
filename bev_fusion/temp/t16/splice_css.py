"""scratch: replace the EXTRA_CSS constant of build_big_report.py by the PAGE_CSS constant of page_css.txt (one-off)"""
p = "bev_fusion/temp/t16/build_big_report.py"
s = open(p, encoding="utf-8").read()
a = s.index('EXTRA_CSS = """')
b = s.index('"""\n', a + len('EXTRA_CSS = """')) + 4
new = open("bev_fusion/temp/t16/page_css.txt", encoding="utf-8").read()
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
print("spliced", len(new), "characters")
