p = "experiment_execution/closed_questions.md"
s = open(p, encoding="utf-8").read()
old = "| a new mechanism: a heat-level replay (two sub-threshold looks summed) with a ceiling >= 0.05 of the GT ends, or a frame rate above the 2 Hz of the keyframes |"
new = "| a learned heat-side state (the noisy-OR replay of step 0c is closed too: net gain 0.009 / 0.008 against the bar 0.05, 1 true to 11.5 false new keypoints, +0.002 over a no-time null) or a frame rate above the 2 Hz of the keyframes |"
assert old in s
s = s.replace(old, new).replace("E299 sections 4 and 5, E336 (closed by data, not run)", "E299 sections 4, 5 and 5c, E336 (closed by data, not run)")
open(p, "w", encoding="utf-8").write(s)
p = "evaluation/README.md"
s = open(p, encoding="utf-8").read()
lines = s.split("\n")
i = next(k for k, l in enumerate(lines) if l.startswith("| `jtime.py` |"))
lines.insert(i + 1, "| `jheat_replay.py` | E299 step 0c (CPU, VAL, read-only): the heat-level replay of a recursive state: the stored E312 model's entry / exit heat of t-1 warped into t and noisy-ORed with the heat of t (w = 1.0 pre-registered, sweep w = 0.5 / 0.25, nulls N1 = heat with itself, N2 = unwarped), keypoints as `JEnds.forward`, scored against the GT ends (net gain share, true-new / false-new, vanished, heat-visible ceiling); `jheat_replay.py --seed S` (about 5 min, 2 threads), `jheat_replay.py selftest`; result: closed by data (card E299 section 5c) |")
open(p, "w", encoding="utf-8").write("\n".join(lines))
