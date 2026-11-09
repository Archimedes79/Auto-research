p = "temp/pitch/patch_r2_v.py"
s = open(p, encoding="utf-8").read()
old = '''b = s.index("class StagesView(BevView):")
tail'''
new = '''b = s.index("    def render(self, sample: Sample, pred: Prediction, ctx: ViewContext) -> Page:", a)
tail'''
assert old in s
s = s.replace(old, new)
old2 = '''        return heat


\'\'\'
s = s[:a] + tail + s[b:]'''
new2 = '''        return heat

\'\'\'
s = s[:a] + tail + s[b:]'''
assert old2 in s
s = s.replace(old2, new2)
open(p, "w", encoding="utf-8").write(s)
