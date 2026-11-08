p = "training/model.py"
s = open(p, encoding="utf-8").read()
a = '''{"camera_off": True}, {"cam_adapter": True}, {"cam_collapse": "zsum"}]'''
b = '''{"camera_off": True}, {"cam_adapter": True}, {"cam_collapse": "zsum"}, {"cam_buffer": "recurrent"}]'''
assert s.count(a) == 1
s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="").write(s)
