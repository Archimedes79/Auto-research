p = "temp/e291/analysis.py"
s = open(p, encoding="utf-8").read()
s = s.replace("    global ROOT\n    old, ROOT = ROOT, tmp\n", "    me = sys.modules[__name__]\n    old, me.ROOT = ROOT, tmp\n")
s = s.replace("    finally:\n        ROOT = old\n", "    finally:\n        me.ROOT = old\n")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("global ROOT" in s)
