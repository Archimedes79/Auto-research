s = open("temp/t26/dev_mut.py").read()
s = s.replace('    a, b, c = run("tj_a"), run("tj_b"), run("tj_c", seed=1)\n', '    a, b, c = run("tj_a"), run("tj_b"), run("tj_c", seed=1)\n    print("DBG", len(a), len(b), len(c), a[3:7])\n')
open("temp/t26/dev_mut.py", "w").write(s)
