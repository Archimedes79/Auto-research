p = "temp/e291/test_e291_staged.py"
s = open(p, encoding="utf-8").read()
s = s.replace("def fake_main(mod, argv, steps=20, record=None):", "def fake_main(mod, argv, steps=20, record=None):\n    n_steps = steps")
s = s.replace("        for s in range(steps):\n            b = next(it)", "        for s in range(n_steps):\n            b = next(it)")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print(s.count("n_steps"))
