import io
p = "temp/e363/verdict_e363.py"
s = io.open(p, encoding="utf-8").read()
head, tail = s.split("def selftest():")
tail = tail.replace('[0] == "TIE"', '[0].startswith("TIE")')
tail = tail.replace('    for fx in (False, None):', '    tok, lines = judge(0, dict(good, a=card(0.700, 0.175, a0)), None, {"a": b_pos}, "n/a", "n/a")\n    assert tok.startswith("TIE") and "attachment-dominated" in tok and any("OUTSIDE" in l for l in lines), tok               # a gain far from the own-keypoint witness is never KEEP\n    tok, lines = judge(0, dict(good, a=card(0.666, 0.175, a0, ent=0.05)), None, {"a": b_pos}, "n/a", "n/a")\n    assert tok == "INVALID" and "(vii)" in lines[-1], (tok, lines[-1])                                                          # the symmetry stop rule of E360 is kept\n    for fx in (False, None):', 1)
io.open(p, "w", encoding="utf-8").write(head + "def selftest():" + tail)
print("ok")
