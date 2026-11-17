p = "temp/e291/test_e291_staged.py"
s = open(p, encoding="utf-8").read()
old = '    root = os.path.join(SCR, "dump_root")\n    os.makedirs(os.path.join(root, "runs"), exist_ok=True)\n'
new = '    import tempfile\n    tmp = tempfile.TemporaryDirectory(prefix="dump_root_", dir=SCR)                                  # a fresh folder per run (the dump refuses an existing read-out folder); removed at the end\n    root = tmp.name\n    os.makedirs(os.path.join(root, "runs"), exist_ok=True)\n'
assert old in s
s = s.replace(old, new)
old2 = '    print("dump ok")\n'
new2 = '    tmp.cleanup()\n    print("dump ok")\n'
assert s.count(old2) == 1
s = s.replace(old2, new2)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
