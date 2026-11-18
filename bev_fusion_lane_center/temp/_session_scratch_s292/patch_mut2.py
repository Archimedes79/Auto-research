import io
p = "temp/_session_scratch_s292/mut_faithful.py"
s = io.open(p, encoding="utf-8").read()
a = ''' ("no junction flag", "    LD.QData.DEFAULT_JUNCTION = True\n    ds2", "    ds2"),'''
assert s.count(a) == 1
s = s.replace(a, ''' ("junction flag not set", "LD.QData.DEFAULT_JUNCTION = True  ", "LD.QData.DEFAULT_JUNCTION = False  "),
 ("junction flag not restored", "LD.QData.DEFAULT_JUNCTION = old_junction", "pass"),
 ("pair loader: no missing assertion", "assert ds2.missing == 0", "assert True"),''')
io.open(p, "w", encoding="utf-8").write(s)
