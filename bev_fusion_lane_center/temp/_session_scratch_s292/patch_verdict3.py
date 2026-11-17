import io
p = "temp/e360/verdict_e360.py"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)
rep("    kp = lambda topo: dict(numbers=dict(head=dict(topo_F1=topo)))  # noqa: E731", "    kp = lambda topo, arm=\"c\": dict(curves=CURVES[0], numbers=dict(head=dict(topo_F1=topo), base=dict(topo_F1=BASE_REF[0][arm])))  # noqa: E731")
rep("ck=kp(0.64), ak=kp(0.65))\n    b_pos", "ck=kp(0.64, \"c\"), ak=kp(0.65, \"a\"))\n    b_pos")
rep("flip = dict(good, ck=kp(0.65), ak=kp(0.64))", "flip = dict(good, ck=kp(0.65, \"c\"), ak=kp(0.64, \"a\"))")
rep('''    wit_bad = dict(good, ck=dict(kp(0.64), curves=CURVES[0], numbers=dict(head=dict(topo_F1=0.64), base=dict(topo_F1=0.1))), ak=dict(kp(0.65), curves=CURVES[0], numbers=dict(head=dict(topo_F1=0.65), base=dict(topo_F1=BASE_REF[0]["a"]))))''',
    '''    wit_bad = dict(good, ck=dict(curves=CURVES[0], numbers=dict(head=dict(topo_F1=0.64), base=dict(topo_F1=0.1))))''')
io.open(p, "w", encoding="utf-8").write(s)
