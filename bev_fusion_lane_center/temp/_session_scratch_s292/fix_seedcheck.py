p = "temp/_session_scratch_s292/patch_fix1.py"
s = open(p, encoding="utf-8").read()
a = '''    for what, path in ((\\"--full\\", full), (\\"--base\\", base)):\n"
     "        assert not path or seed is None or re.search(rf\\"_s{seed}(?!\\d)\\", str(path)), f\\"{what} {path}: its path does not carry _s{seed}: the dump and the checkpoint must be of seed {seed}\\"\n"),'''
assert s.count(a) == 1, s.count(a)
b = '''    for what, path, name in ((\\"--full\\", full, os.path.basename(str(full))), (\\"--base\\", base, os.path.basename(os.path.dirname(str(base))))):      # the file name of the dump, the run folder of the checkpoint\n"
     "        assert not path or seed is None or re.search(rf\\"_s{seed}(?!\\d)\\", name), f\\"{what} {path}: its name does not carry _s{seed}: the dump and the checkpoint must be of seed {seed}\\"\n"),'''
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(a, b))
print("ok")
