import os
def sub(path, a, b):
    s = open(path, encoding="utf-8", newline="").read()
    assert s.count(a) == 1, (path, s.count(a))
    s = s.replace(a, b)
    tmp = path + ".tmp_gate"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(s)
    os.replace(tmp, path)
sub("evaluation/jends_eval.py", 'gpu_gate.wait_for_room(who="jends_eval dump")', 'gpu_gate.wait_for_room(who=f"jends_eval dump {a.run}")')
sub("evaluation/jmaps_eval.py", 'gpu_gate.wait_for_room(who="jmaps_eval score")', 'gpu_gate.wait_for_room(who=f"jmaps_eval score {a.run}")')
print("ok")
