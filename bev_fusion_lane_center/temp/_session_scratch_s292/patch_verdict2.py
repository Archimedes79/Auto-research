import io, re
p = "temp/e360/verdict_e360.py"
s = io.open(p, encoding="utf-8").read()
head, tail = s.split("def selftest():")
tail = re.sub(r"(judge\(0, [a-z_]+, None, \{\"a\": [a-z_0-9]+\}), None\)", r'\1, "n/a")', tail)
tail = tail.replace('''    assert interval("/nonexistent/x.log") is None''', '''    tok, lines = judge(0, good, None, {"a": b_pos}, None)
    assert tok == "INVALID" and "(v)" in lines[-1], (tok, lines[-1])                                      # arm t missing on seed 0: a failed stop rule
    full_base = lambda t: dict(topo_F1=t, F1_x=0.8)  # noqa: E731
    stored = dict(c=dict(topo_F1=BASE_REF[0]["c"], vert100=19.0), a=dict(topo_F1=BASE_REF[0]["a"], vert100=19.0))
    good2 = dict(c=dict(good["c"], numbers=dict(head=good["c"]["numbers"]["head"], base=dict(topo_F1=BASE_REF[0]["c"], vert100=19.0))),
                 a=dict(good["a"], numbers=dict(head=good["a"]["numbers"]["head"], base=dict(topo_F1=BASE_REF[0]["a"], vert100=19.0))))
    tok, _ = judge(0, good2, stored, {"a": b_pos}, "n/a")
    assert tok == "TIE", tok                                                                             # whole base block equal, no witness: TIE
    stored_bad = dict(stored, a=dict(topo_F1=BASE_REF[0]["a"], vert100=25.0))
    tok, lines = judge(0, good2, stored_bad, {"a": b_pos}, "n/a")
    assert tok == "INVALID" and "vert100" in lines[-1], (tok, lines[-1])                                 # one key of the base block differs
    wit_bad = dict(good, ck=dict(kp(0.64), curves=CURVES[0], numbers=dict(head=dict(topo_F1=0.64), base=dict(topo_F1=0.1))), ak=dict(kp(0.65), curves=CURVES[0], numbers=dict(head=dict(topo_F1=0.65), base=dict(topo_F1=BASE_REF[0]["a"]))))
    tok, lines = judge(0, wit_bad, None, {"a": b_pos}, "n/a")
    assert tok == "INVALID" and "(i/ii) ck" in lines[-1], (tok, lines[-1])                               # a witness on another base block
    assert interval("/nonexistent/x.log") is None''')
tail = tail.replace('print("verdict_e360 selftest ok (10 planted cases)")', 'print("verdict_e360 selftest ok (14 planted cases)")')
io.open(p, "w", encoding="utf-8").write(head + "def selftest():" + tail)
print("ok")
