p = "temp/jr/fail_gallery.py"
s = open(p, encoding="utf-8").read()
old = 'prs = [(e["kp_pos"][i], e["kp_dir"][i], e["kp_pos"][k + j], e["kp_dir"][k + j]) for i, j in zip(ia, ib)]'
assert old in s
s = s.replace(old, 'prs = [(e["kp_pos"][i], e["kp_dir"][i], e["kp_pos"][k + j], e["kp_dir"][k + j]) + (tuple(e["scale"][i, j]) if e.get("scale") is not None else ()) for i, j in zip(ia, ib)]', 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
