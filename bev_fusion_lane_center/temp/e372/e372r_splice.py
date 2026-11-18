"""throw-away: replace test_live of the scratch test copy by temp/e372/e372r_test_live_new.txt (the file's own line endings kept)"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(HERE, "e372r_test_jtoken_eval.py")
src = open(path, newline="").read()
nl = "\r\n" if "\r\n" in src else "\n"
new = open(os.path.join(HERE, "e372r_test_live_new.txt"), newline="").read().replace("\r\n", "\n").rstrip("\n").replace("\n", nl)
a, b = src.index("def test_live():"), src.index("TESTS = [")
assert src.count("def test_live():") == 1 and a < b
open(path, "w", newline="").write(src[:a] + new + nl + nl + nl + src[b:])
print("spliced", len(new.splitlines()), "lines")
