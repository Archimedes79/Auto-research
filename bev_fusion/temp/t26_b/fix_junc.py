p = "evaluation/bev_junction.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    res = {k: m.result()[0.0]["lane_divider"]["all"]["all"] for k, m in meters.items()}
''', '''    res = {k: m.result()[0.0]["lane_divider"]["all"]["all"] for k, m in meters.items()}
    assert on_road[0] / len(done) > max(on_road[1:]) / len(done) + 0.2, "orientation: the mask must lie on the drivable area, a flipped one does not"
''')
open(p, "w", encoding="utf-8").write(s)
