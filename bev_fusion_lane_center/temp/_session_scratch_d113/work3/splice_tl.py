import os
W = os.path.join(os.environ["BFLC_ROOT"], "temp/_session_scratch_d113/work3")
s = open(os.path.join(W, "patch_tl.py"), encoding="utf-8").read()
more = open(os.path.join(W, "more_tl.txt"), encoding="utf-8").read()
assert "@@MORE@@" in s
open(os.path.join(W, "patch_tl.py"), "w", encoding="utf-8").write(s.replace("@@MORE@@", more))
print("spliced")
