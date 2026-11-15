p = "temp/t26/edit_test2.py"
s = open(p).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)
rep('''all(a[i][0] != a[i + 1][0] for i in range(4, 15, 2)) and len({x[0] + [x[1]] and tuple(x[0]) for x in a[4:]}) == 12''', '''all(a[i][0] != a[i + 1][0] for i in range(4, 15, 2)) and len({(tuple(x[0]), x[1]) for x in a[4:]}) == 12''')
rep('''        real, calls = d._image_uint8, [0]

        def flaky(sd):
            calls[0] += 1
            if calls[0] == 3:
                raise RuntimeError("disk full")
            return real(sd)''', '''        real, calls, n_ok = d._image_uint8, [0], (2 if older else 1)                       # two older images per sample, one key frame

        def flaky(sd):
            calls[0] += 1
            if calls[0] == n_ok + 1:
                raise RuntimeError("disk full")
            return real(sd)''')
rep('''        made = {os.path.join(dp, f): None for dp, _, fs in os.walk(folder) for f in fs if f.endswith(".jpg") and (older or SWEEPS not in dp)}
        made = {k: os.path.getmtime(k) for k in made}''', '''        made = {os.path.join(dp, f): os.path.getmtime(os.path.join(dp, f)) for dp, _, fs in os.walk(folder) for f in fs if f.endswith(".jpg") and (older or SWEEPS not in dp)}''')
rep('''crashes on its 3rd image leaves no marker''', '''crashes on its image after the first sample leaves no marker''')
rep('''died and marker == {} and len(made) == 2 and bool(marker2)''', '''died and marker == {} and len(made) == n_ok and bool(marker2)''')
rep('''kcfg = dataclasses.replace(cfg, train_scenes=tuple(s1)[:1], val_scenes=tuple(s1)[:1])''', '''kcfg = dataclasses.replace(cfg, train_scenes=(scene_of(toks[0]),), val_scenes=(scene_of(toks[0]),))''')
open(p, "w").write(s)
