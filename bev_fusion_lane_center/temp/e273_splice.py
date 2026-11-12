"""one-off: put revision 4 (port graph) into the E273 card: section 2.2 replaced, the anchor wording elsewhere adjusted"""
p = "experiments/E273_pyramid-head-soft-graph-map-assignment/experiment.md"
s = open(p, encoding="utf-8").read()
a, b = s.index("### 2.2 "), s.index("### 2.3 ")
s = s[:a] + open("temp/e273_s22.md", encoding="utf-8").read() + s[b:]
rep = [
    ("# E273 Pyramid head -> lane-piece decoder -> map lane graph", "# E273 Pyramid head -> port graph (pixels, then smooth connections) -> map lane graph"),
    ('"title": "pyramid head + lane-piece decoder (one query per lane piece, matching-free anchors) + map lane graph encoder',
     '"title": "pyramid head (centre and port pixels) + port graph (smooth connections between ports, one id per piece) + map lane graph encoder'),
    ("-> section 1b (revision 2 of this card, same day: the vertex-level soft graph of revision 1 is replaced by a lane-piece decoder).",
     "-> section 1b (revision 2 of this card, same day: the vertex-level soft graph of revision 1 is replaced by a lane-piece decoder; revision 3: intersections; revision 4: Mirko's two layers, centre and port pixels, then smooth connections between ports, replace the anchors)."),
    ("[P] x [E] --> [A] assignment", "[G] x [E] --> [A] assignment"),
    ("| **pyramid head + lane-piece decoder (this card)** |", "| **pyramid head + port graph (this card, revision 4)** |"),
]
for o, n in rep:
    assert o in s, o[:60]
    s = s.replace(o, n)
i = s.index("- New at 1 m for the graph:")
j = s.index("\n", s.index("- New at 0.5 m:"))
s = s[:i] + ("- New at 1 m for the graph (layer 1 of section 2.2): 8 heading-binned centre heats (a cell can be on in two bins: crossing and diverging pieces are separable, "
             "the 1.9 % / 12.1 % cells of section 1) and the port heats with offset and heading (split, merge, lane start, lane end / dead end, window edge); "
             "key-point heads of this kind gave split / merge F1 0.55 / 0.61 in E44.\n- New at 0.5 m: the stop line (autogt `stop_line`), the strongest visible cue where a lane enters a junction.") + s[j:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
