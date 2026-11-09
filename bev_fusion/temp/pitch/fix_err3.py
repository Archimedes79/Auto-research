p = "evaluation/bev_errors.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
rep('''from evalsuite.comparison import boxes as BX  # noqa: E402
''', '''from evalsuite.comparison import boxes as BX  # noqa: E402
from evaluation.metrics import in_front  # noqa: E402
''')
rep('''        dets, gts = boxes[(scene, k)]
        r = classify_frame(dets, gts, groups, thr, min_score)''', '''        dets, gts = boxes[(scene, k)]
        r = classify_frame([d for d in dets if in_front(d)], [g for g in gts if in_front(g)], groups, thr, min_score)      # the scored window of the metric (x < 60 m, +-32.3 deg)''')
rep('''box("vehicle", 60.0, 20.0, 0.95),                  # a ghost: FP isolated''', '''box("vehicle", 55.0, 20.0, 0.95),                  # a ghost: FP isolated''')
rep('''Input: the boxes of every frame''', '''Only boxes inside the scored window of the metric (x < 60 m, +-32.3 deg of the front axis) are counted, as the numbers do.
Input: the boxes of every frame''')
# the window: a box outside is no error
rep('''        miss = [x for x in rows if x["scene"] == "scene-1" and x["index"] == 1]''', '''        assert all(x["x"] < 60 and abs(math.atan2(x["y"], x["x"])) <= 0.5637 for x in rows), "only the scored window"
        miss = [x for x in rows if x["scene"] == "scene-1" and x["index"] == 1]''')
open(p, "w", encoding="utf-8").write(s)
