p = "evaluation/bev_views.py"
s = open(p, encoding="utf-8").read()
old = "Bird's-eye maps are drawn with x forward (up) and +y (left of the car) on the left; arrays are (x, y) = (rows, columns).\n"
new = old + ("Colour scheme (Mirko, 2026-11-03): black plot background, dark-grey road with a black boundary, white lane lines, light-grey crossings, a calmer green for walkways, dark-red\n"
             "objects (vehicles, pedestrians, two-wheelers) and the same red translucent where they lie outside the observed area (camera x radar x not hidden); boxes: GT outline red\n"
             "(translucent outside), the dashed GT outline on a prediction white, predicted boxes amber. The error maps (right grey, false red, missed blue, cream) are unchanged.\n")
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
