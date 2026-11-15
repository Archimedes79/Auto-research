p = "evaluation/jmaneuver.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:60]
    s = s.replace(old, new, 1)


sub("    shifted = dict(pts=[arc + np.array([3.0, 0.0])])                                  # 3 m aside: not realised\n",
    "    c = np.array([-15.0, -15.0])                                                       # the centre of the circle: radial offsets keep the direction and move the lane sideways everywhere\n    shifted = dict(pts=[c + (arc - c) * (19.0 / 15.0)])                               # 4 m aside: not realised\n")
sub("    near = dict(pts=[arc + np.array([0.5, 0.3])])                                      # 0.6 m aside: realised\n", "    near = dict(pts=[c + (arc - c) * (15.6 / 15.0)])                                   # 0.6 m aside: realised\n")
sub("(the shifted copy lies 3 m away: none)", "(the shifted copy lies 4 m away: none)")
open(p, "w", encoding="utf-8").write(s)
