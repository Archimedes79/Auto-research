p = "evaluation/bev_gtcheck.py"
t = open(p, encoding="utf-8").read()
a = '''            if side.get("steps"):                                       # steps: the ego motion from cur to next, from prev to cur (what the time mover is given)
                a, b = (s[0.0], s[dt]) if dt > 0 else (s[dt], s[0.0])
                px, py, pyaw = rel_pose(a["meta"]["T_e2g"], b["meta"]["T_e2g"])'''
b = '''            if side.get("steps"):                                       # steps: the ego motion from cur to the neighbour, backwards in time for prev (what the time mover is given)
                px, py, pyaw = rel_pose(s[0.0]["meta"]["T_e2g"], s[dt]["meta"]["T_e2g"])'''
assert t.count(a) == 1
t = t.replace(a, b)
open(p, "w", encoding="utf-8", newline="\n").write(t)
print("patched")
