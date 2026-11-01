import glob
def sub(path, pairs):
    t = open(path, encoding="utf-8").read()
    for a, b in pairs:
        assert a in t, (path, a[:60])
        t = t.replace(a, b)
    open(path, "w", encoding="utf-8").write(t)
sub(glob.glob("experiments/E240*/experiment.md")[0], [
    ("are 10.1 (lanes found) and 7.7 (R@1 far) paired SDs", "are 10.1 (lanes found) and 7.6 (R@1 far) paired SDs"),
    ("the Q edge is positive in all four pairs but small and, in three of four pairs, below the band.", "the Q edge is positive in all four pairs (+0.0062, standard error 0.0022, t about 2.9 on 3 df) but small and, in three of four pairs, below the band: below the 0.010 decision band, not shown to be zero."),
])
sub(glob.glob("experiments/E255*/experiment.md")[0], [
    ("while the controls lose only 0.006 (0.731 -> 0.724) over the same squeeze.", "while the controls lose 0.007 (0.731 -> 0.724) over a smaller squeeze (15.1 -> 14.3 vertices): per vertex removed the Poisson arm pays about twice as much (0.016 against 0.0085 Q)."),
    ("so thinning removes true lanes, not duplicates.", "our reading is that thinning removes true lanes (lanes found 0.640 -> 0.551); no score-calibration analysis was made."),
])
sub(glob.glob("experiments/E256*/experiment.md")[0], [
    ("- **Result: the ranking is confirmed, no pair flips**", "- **Result: no pair flips, the order is not established** (the rule can hardly fire: the whole Singapore spread is 0.008, below the 0.010 flip threshold, so \"prediction hit\" is weak evidence)"),
])
