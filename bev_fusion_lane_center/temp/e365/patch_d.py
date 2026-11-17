import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)
rep("score --run R --seed S [--split val|train] [--drop 0.17 --drop-seed 0]  ", "score --run R --seed S [--split val|train] [--drop 0.17 --drop-seed 0] [--max N]  ")
rep("runs/eval/e365_ctl/<arm>_<ident>_s<S>.json", "runs/eval/e365_ctl/<arm>_<ident>_s<S>_K<K>.json")
rep("section 0): no GT field of the GT table enters" if False else "no field of the GT table enters (`check_writer` asserts it on every frame)", "no field of the GT table enters (the real guarantee: the head reads only `jtoken.INPUTS`, test_jtoken.py; `check_writer` / `frame_prediction` assert on every frame that the nodes are the kept tokens' own, e = 1 - p_none and the links are the dump's)")
open(p, "w", encoding="utf-8").write(s)
