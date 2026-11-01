import os
os.chdir("/mnt/c/Auto-research")
R = lambda p: open(p, encoding="utf-8").read()
W = lambda p, s: open(p, "w", encoding="utf-8", newline="\n").write(s)
c = R("autogt/cli.py")
c = c.replace("""    from . import sets
    want = sets.scenes(a.set_name) if a.set_name else _scene_names(a)""", """    from . import current_version, sets
    want = sets.scenes(a.set_name) if a.set_name else _scene_names(a)""", 1)
c = c.replace("""f"{__import__('autogt').current_version()}; missing {len(r['missing'])}")""", """f"{current_version()}; missing {len(r['missing'])}")""", 1)
W("autogt/cli.py", c)
t = R("autogt/selftest.py")
old = "        gen5 = _Gen2(root, \"v1.0-mini\", ss=4, light=True, scenes=[\"scene-0061\"])"
assert old in t
new = '''        want_ = sets.scenes("small")[:2]                                               # `have`: the check before a training
        st2 = sets.stored_for(want_, data=d, version="v1.0-mini", dataroot=root)
        assert st2["total"] == len(st2["missing"]) + st2["have"] and st2["have"] > 0 and len(st2["missing"]) > 0 and st2["scenes"] == 2, st2
        stored_t = {t_ for t_ in small_tokens if os.path.exists(os.path.join(sd5, t_ + ".npz"))}
        assert st2["have"] == len(stored_t & set(sets.tokens("small", "v1.0-mini", root)) & set(
            t_ for s_ in want_ for t_ in __import__("autogt").scene_tokens(s_, "v1.0-mini", root))), st2
        assert sets.stored_for(want_, data=d, version="v1.0-mini", dataroot=root, stride=1000)["total"] == 2, "stride"
'''
t = t.replace(old, new + old, 1)
W("autogt/selftest.py", t)
print("ok")
