def sub(p, pairs):
    s = open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (p, a[:60])
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8", newline="\n").write(s)
sub("experiments/E337_attached-end-geometry-gap/experiment.md", [
    ("(the stored cards' `ends_within_1m_of_a_delivered_piece_end` field is a different population, the ends the connectors use: attached 0.383 / 0.360 against keypoint 0.374 / 0.349 on seed 0 / 1 entry / exit as stored)",
     "(the stored field `ends_within_1m_of_a_delivered_piece_end` of the PRIMARY cards is a different population, the ends the connectors use, entry / exit: seed 0 attached 0.383 / 0.435, keypoint 0.374 / 0.427; seed 1 attached 0.360 / 0.444, keypoint 0.349 / 0.433)"),
    ("(E324 section 5 had 24 % / 26 % from rounded inputs: corrected there)", "(E324 section 5 had +0.0024 and 26 % from rounded inputs: corrected there)"),
])
sub("experiments/E324_e320-duplicates-refined/experiment.md", [
    ("the attached ends +0.0024 / +0.0032 over the own keypoints in the nogeom read-out (a - kn): the learned attachment recovers a quarter to a third of what the snap's geometry gives (26 % / 34 %).",
     "the attached ends +0.0023 / +0.0032 over the own keypoints in the nogeom read-out (a - kn): the learned attachment recovers a quarter to a third of what the snap's geometry gives (25 % / 34 %; unrounded +0.002348 / +0.009281 and +0.003222 / +0.009378). The gap is geometry, not wiring: the attached end and the own keypoint wire the same delivered piece for 96 % of the ends (E337)."),
    ("see `experiments/E337_attached-end-geometry-gap/experiment.md` (the PRIMARY read-out gives up +0.0069 / +0.0062 topo F1 against the snap geometry; step 0 on the stored dumps before anything is trained).",
     "see `experiments/E337_attached-end-geometry-gap/experiment.md` (the PRIMARY read-out gives up +0.0069 / +0.0062 topo F1 against the snap geometry; answered from the stored cards and one scan of the dumps: geometry, not wiring; the sharpening lever is a stub, not scheduled)."),
])
print("ok")
