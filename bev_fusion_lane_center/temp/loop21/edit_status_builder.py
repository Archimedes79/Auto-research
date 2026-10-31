p = "reports/build/reports/build_status21_report.py"
s = open(p, encoding="utf-8").read()
R = [
 ("the camera is for orientation only, the system never sees it as an image: it uses the BEV features.", "the camera image is for orientation: the lane project itself works on the cached BEV features that the frozen foundation computed from camera and radar."),
 ("Lane heat and heading on the grid (U-Net-type head). It is the input of the builder and of the map registration; it is a pixel-based head.", "Lane heat and heading on the grid (U-Net / FCN-type head). It is the input of the map registration and a feature of the builder; it is a pixel-based head."),
 ("U-Net (K62); direction field with heat as in dense lane detectors", "U-Net / FCN (K62)"),
 ("each seed reads the features along 12 rays with ≤5 rings (bilinear sampling, no argmax)", "each seed reads the features along a teardrop of 12 rays with ≤5 rings (loop-16 design, bilinear sampling, no argmax)"),
 ("Loss and threshold", "Loss and threshold"),
 ("(blue = upward, magenta / red = left-up)", "(opposite directions get different hues)"),
 ("only seeds with path validity above tau 0.7 (coloured lines) survive; the other ~60 % (grey) are discarded.", "only nodes with path validity above tau 0.7 (coloured lines, 237 of 576 here) are used; the rest (grey) is discarded."),
 ("the junction head marks many ground-truth forks (5) but with weaker peaks than in the Singapore example;", "the junction head (5) has heat near the ground-truth forks, several of them weak;"),
 ("(1) on straight and gently curved roads the graph is almost exact and uses 24 - 60 vertices where the control uses 62 - 225;", "(1) on straight and gently curved roads the graph is almost exact; in these six frames recall_dp has 24 - 85 vertices and the control 62 - 225 (the control is not simplified);"),
 ("the third frame (truck ahead, GT behind it) is one where recall_dp beats the control (0.81 against 0.65).", "the third frame (a truck and a car ahead) is one where recall_dp beats the control (0.81 against 0.65)."),
 ("in the curve (bottom) the left lane edge is stretched and a parallel duplicate appears.", "in the curve (bottom) the lines are partly doubled."),
 ("it loses 0.1 Q at 12 vertices;", "it loses 0.12 Q at 12 vertices;"),
 ("the seed SD is 0.0025, so recall over champion is about two SDs, champion against control is a tie.", "the seed SD is 0.0025; recall over champion is +0.0054, but only +0.0025 at a matched vertex budget (E236); champion against control is a tie."),
]
for a, b in R:
    if a == b:
        continue
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
