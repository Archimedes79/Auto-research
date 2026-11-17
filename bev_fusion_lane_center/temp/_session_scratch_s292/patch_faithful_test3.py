import io
p = "evaluation/jwire_data.py"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:90], s.count(a))
    s = s.replace(a, b)
rep("    rows = [10, 11, 12, 13, 14, 15]\n    calls, seen = [], []\n", "    rows = [10, 11, 12, 13, 14, 15]\n    calls, seen, pcalls = [], [], []\n")
rep('''                 qpredict=NS(predict=lambda mb, b: [dict(nodes=dict(pos=float(b["x"][i, 0]), dirv=0, e=0, flags=0, section=0), links=dict(src=0, dst=0, p=0, long=0, a=0, b=0)) for i in range(len(b["row"]))]))''',
    '''                 qpredict=NS(predict=lambda mb, b: pcalls.append(len(b["row"])) or [dict(nodes=dict(pos=float(b["x"][i, 0]), dirv=0, e=0, flags=0, section=0), links=dict(src=0, dst=0, p=0, long=0, a=0, b=0)) for i in range(len(b["row"]))]))''')
rep('''        assert DUMP_B == dict(bs=4, map_mode="exact")''', '''        assert pcalls == [4, 2], pcalls                                                                        # B's loader was read until nmax = 5 frames were there (two batches)
        pcalls.clear()
        r4 = live_records(0, "val_clean", 0.1, 4, device="cpu", pair_run="e324_s_s0", ends="att", faithful=True)
        assert [r["row"] for r in r4] == rows[:4] and pcalls == [4], (pcalls, "B's loader stops as soon as nmax frames are there")
        calls[:] = calls[:2]
        assert DUMP_B == dict(bs=4, map_mode="exact")''')
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
