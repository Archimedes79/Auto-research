import re
p = "temp/agt_gallery/build_overview_json.py"
s = open(p, encoding="utf-8").read()
reps = [
 ("close-ups at full resolution", "close-ups at full resolution"),
 ("7 full-resolution close-ups: lines on the paint within about 0.1-0.3 m in Boston and Singapore, day and rain; centre-line and divider cells",
  "7 full-resolution close-ups: in the 3 Boston ones with visible paint (scene-0094 day, 0200 cloudy, 0441 heavy rain) the lines lie on the paint within about 0.1-0.3 m; the Singapore close-ups show map arcs and no paint to compare; centre-line and divider cells"),
 ("the 4 Boston close-ups seen here show none", "the Boston close-ups seen here show none"),
 ("map lines lie on\",\n         \"  the paint within a few decimetres,", "map lines lie on\",\n         \"  the paint within a few decimetres (Boston close-ups),"),
 ("(18 four-panel examples and 7 close-ups were inspected one by one, the thumbnails as a whole)", "(21 four-panel examples and 7 close-ups were inspected one by one, the thumbnails as a whole)"),
 ("Map lines follow the road geometry in every close-up; where paint is visible (dashed lane divider in Boston, faded double yellow in rain, the road centre line) the line lies",
  "Map lines follow the road geometry in every close-up; where paint is visible (3 Boston close-ups: dashed lane divider, faded double yellow in rain, the road centre line) the line lies"),
]
for a, b in reps:
    if a not in s:
        print("NOT FOUND:", a[:70])
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
