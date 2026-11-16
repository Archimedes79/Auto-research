p = "temp/jr/build_junction_report.py"
s = open(p, encoding="utf-8").read()
old = '''             ("E314: the same on the sigma 0.7 m maps of E313 (TIE with E312)", "runs/e314_p_s{s}/ends_card_snap_thr0.1.json")]'''
assert old in s
s = s.replace(old, '''             ("E314: the same on the sigma 0.7 m maps of E313 (TIE with E312)", "runs/e314_p_s{s}/ends_card_snap_thr0.1.json"),
             ("E298: E312 with the driven pairs weighted higher (the planner variant; driven pair recall +0.12 / +0.11)", "runs/e298_w_s{s}/ends_card_snap_thr0.1.json")]''', 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
