p = "tests/radar_codecs.py"
s = open(p, encoding="utf-8").read()
body = open("temp/radar/_scratch_patches/level_cv.txt", encoding="utf-8").read()
s = s.replace("def level_e(steps, names=(\"abs\", \"delta\")):", body + "def level_e(steps, names=(\"abs\", \"delta\")):", 1)
s = s.replace('''    if "f" in sys.argv[2:]:
        level_f(steps)''', '''    if "cv" in sys.argv[2:]:
        level_cv(steps, seeds=tuple(range(int(sys.argv[3]) if len(sys.argv) > 3 else 2)))
    elif "f" in sys.argv[2:]:
        level_f(steps)''', 1)
s = s.replace("leaky + identity start). Results: tests/README.md and plan.md section 3.4.", "leaky + identity start); cv = 5-fold cross-validation over scenes, paired codec comparison with a written rule (arg 3 = seeds per fold).\nResults: tests/README.md and plan.md section 3.4.", 1)
s = s.replace("[steps] [e | f]", "[steps] [e | f | cv [seeds]]", 1)
open(p, "w", encoding="utf-8").write(s)
print(s.count("level_cv"))
