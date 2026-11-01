import sys
S = "temp/radar/_scratch_patches/"
plan_new = ("so the learned part overfits by about 0.5 point. Cross-validation over scenes (run `r1.codec_cv`, 5 folds of 2 scenes, 1 seed, 1000 steps, probe with pass-through start, tolerances x 0.25): kept / oracle abs 74.9 +- 5.5 %, delta 95.4 +- 1.4 %, phys 96.4 +- 1.3 %; paired delta - abs +20.5 +- 4.9 points (5 of 5 folds, t = 9.4): delta better; phys - delta +1.0 +- 0.2 points, inside the 2-point band: TIE (phys needs fewer channels but stores no dr, dpsi, so its features carry no sensor-relative geometry: delta kept); at tolerances x 1 all three are within 1 point (TIE). Overfitting excess (probe gap minus the oracle's own gap): +0.6 points for delta and phys, +1.9 for abs (worst fold +5.2). Caveats: one seed per fold (the fold SD contains the seed noise, a separate seed SD was not measured) and 1000 steps (abs at 2000 steps reached 81.2 / 75.9 % on the 8 / 2 split, so part of its deficit is slower convergence).")
readme_new = ("so overfitting is about 0.5 point; cv (run `r1.codec_cv`, 5 scene folds, 1 seed, 1000 steps, tol x 0.25): kept / oracle abs 74.9 +- 5.5 %, delta 95.4 +- 1.4 %, phys 96.4 +- 1.3 %, paired delta - abs +20.5 +- 4.9 points (5/5 folds) = delta better, phys - delta +1.0 inside the band = TIE; tol x 1 all within 1 point; overfitting excess +0.6 (delta, phys), +1.9 (abs) |")
for name, files, old, new in (("plan", ("plan_head.md", "../../../plan.md"), "so the learned part overfits by about 0.5 point.", plan_new),
                              ("readme", ("readme_head.md", "../../../tests/README.md"), "so overfitting is about 0.5 point |", readme_new)):
    for f in files:
        path = S + f
        t = open(path, encoding="utf-8").read()
        assert t.count(old) == 1, (path, t.count(old))
        open(path, "w", encoding="utf-8").write(t.replace(old, new))
print("ok")
