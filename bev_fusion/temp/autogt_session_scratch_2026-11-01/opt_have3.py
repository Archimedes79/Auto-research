import os
os.chdir("/mnt/c/Auto-research")
t = open("autogt/selftest.py", encoding="utf-8").read()
a = t.index("        want_ = sets.scenes(\"small\")[:2]")
b = t.index("        gen5 = _Gen2(")
new = '''        from .suite import scene_tokens as tokens_of                                   # `have`: the check before a training
        want_ = sets.scenes("small")[:2]
        every = sets.stored_for(want_, data=d, version="v1.0-mini", dataroot=root)
        assert every["missing"] == [] and every["have"] == every["total"] > 0 and every["scenes"] == 2, every
        gone = tokens_of(want_[0], "v1.0-mini", root)[0]
        os.remove(os.path.join(sd5, gone + ".npz"))
        gap = sets.stored_for(want_, data=d, version="v1.0-mini", dataroot=root)
        assert gap["missing"] == [gone] and gap["have"] == gap["total"] - 1, gap
        open(os.path.join(sd5, gone + ".npz"), "wb").write(b"x")
        assert sets.stored_for(want_, data=d, version="v1.0-mini", dataroot=root, stride=1000)["total"] == 2, "stride"
'''
t = t[:a] + new + t[b:]
open("autogt/selftest.py", "w", encoding="utf-8", newline="\n").write(t)
print("test fixed")
