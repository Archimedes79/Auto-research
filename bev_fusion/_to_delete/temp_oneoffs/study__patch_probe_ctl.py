p = "temp/calib/probe_model_stages.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''run = args[0]''', '''control = ""
if "--control" in args:
    i = args.index("--control")
    control = args[i + 1]
    del args[i:i + 2]
run = args[0]''', 1)
s = s.replace('''            CAP.clear()
''', '''            CAP.clear()
            if control:
                ev.perturb(batch, control)
''', 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
