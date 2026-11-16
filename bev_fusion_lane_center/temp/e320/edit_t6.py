p = "temp/e320/t_real.py"
s = open(p, encoding="utf-8").read()
old = '''        jm = modsrc(folder, "jmaneuver")
        jm.score('''
new = '''        jm = modsrc(folder, "jmaneuver")
        QGE.val_frames = orig_vf                                                                              # jmaneuver reads the full B dump itself: all 873 frames, --max 8 stops it
        jm.score('''
assert s.count(old) == 1
s = s.replace(old, new)
old2 = '''        res[(folder, "man")] = json.load(open(os.path.join(d, "maneuver_val.json")))["variants"]'''
new2 = '''        res[(folder, "man")] = json.load(open(os.path.join(d, "maneuver_val.json")))["variants"]
        QGE.val_frames = lambda: orig_vf()[:8]'''
assert s.count(old2) == 1
s = s.replace(old2, new2)
open(p, "w", encoding="utf-8").write(s)
