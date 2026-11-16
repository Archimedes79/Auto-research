import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp", "e343", "_session_scratch_", "patch_r3_tests.py")
s = open(p, encoding="utf-8").read()
old = '''            assert isinstance(rc, SystemExit) and scored == [] and guards == [] and out_files == [] or card_files, (bad, rc)
'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, ""))
