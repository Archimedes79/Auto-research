p = "temp/loop21/make_results_video.py"
t = open(p, encoding="utf-8").read()
for a, b in (("W, H = 1280, 720", "W, H = 1328, 720"),
             (".resize((640, 720))", ".resize((688, 720))"),
             ('"Lane centre-line graph from camera + radar"', '"Lane graph from camera + radar"'),
             ('f"system: {name} (registry), VAL scene {n + 1}/{len(ix)}"', 'f"system: {name} (registry name), VAL scene {s}"')):
    assert a in t, a
    t = t.replace(a, b)
open(p, "w", encoding="utf-8").write(t)
