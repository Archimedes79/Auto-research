import io
def edit(p, pairs):
    s = io.open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (p, a[:90], s.count(a))
        s = s.replace(a, b)
    io.open(p, "w", encoding="utf-8").write(s)
edit("evaluation/jwire_data.py", [
 ("    reverse = []   ", "    miss = []                                                                                              # the pair loader's dataset has frames without stored lane GT\n    reverse = []   "),
 ("        return NS(missing=0), make(bs, len(calls) == 2)", "        return NS(missing=1 if (len(calls) == 2 and miss) else 0), make(bs, len(calls) == 2)"),
 ("        reverse.clear()\n        calls.clear()\n        pcalls[:] = [4, 2]", "        reverse.clear()\n        calls.clear()\n        miss.append(1)\n        assert \"no stored lane GT\" in (refusal(live_records, 0, \"val_clean\", 0.1, 5, device=\"cpu\", pair_run=\"e324_s_s0\", ends=\"att\", faithful=True) or \"\"), \"the pair loader's dataset is checked like the dump does\"\n        miss.clear()\n        assert (QData.DEFAULT_ENDS, QData.DEFAULT_JUNCTION) == (False, False), \"the flags are put back also when the pair pass is refused\"\n        calls.clear()\n        pcalls[:] = [4, 2]"),
])
p = "temp/_session_scratch_s292/mut_faithful.py"
s = io.open(p, encoding="utf-8").read()
a = ''' ("pair loader map mode = B's", 'map_mode=DUMP_PAIR["map_mode"], pixels=False)\n    by_row', 'map_mode=DUMP_B["map_mode"], pixels=False)\n    by_row'),'''
assert s.count(a) == 1, s.count(a)
s = s.replace(a, ''' ("pair loader map mode = B's", 'map_mode=DUMP_PAIR["map_mode"], pixels=False)', 'map_mode=DUMP_B["map_mode"], pixels=False)'),''')
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
