p = "evaluation/soa_maps.py"
s = open(p, encoding="utf-8").read()
s = s.replace('("road_divider", {(98, 51), (99, 51)})', '("road_divider", {(ix, 51) for ix in range(98, NX)})')
s = s.replace("# (99, 51): the clip artefact of the old region", "# rows 99 .. 119 of the last column: the clip artefact of the old region")
s = s.replace("ix 99 reads it too (clipped)", "every row beyond reads it too (clipped)")
open(p, "w", encoding="utf-8").write(s)
