"""one-off: revision 4 of the E273 card, second pass: the lines that still described the anchors of revision 2 / 3"""
p = "experiments/E273_pyramid-head-soft-graph-map-assignment/experiment.md"
s = open(p, encoding="utf-8").read()
rep = [
    ("it starts from anchors the dense maps propose and refines them on the BEV features.",
     "it starts from ports the dense maps propose and checks smooth connections between them against the dense evidence (revision 4, section 2.2)."),
    ("**Matching-free training (the project rule: no matching):** CenterNet / CondLaneNet style. Each GT piece has ONE anchor, the cell of its arclength midpoint (midpoints of the two branches of a split are metres apart, unlike their start points); the 1 m head predicts an anchor heat map; in training the queries are taken at the GT anchors plus at predicted peaks (an index choice), and each query's target is the GT piece whose anchor it sits on (by location, no assignment problem to solve); queries off every anchor learn \"no piece\". Duplicates are what the anchor heat map's peak shape and a learned existence score handle, as in CenterNet.",
     "**Matching-free training (the project rule: no matching):** ports are key points (CenterNet style: heat maps, targets by location); a candidate connection is positive when its two ports are the two ends of one GT piece (by location, no assignment problem to solve). Revisions 2 and 3 used one anchor per piece in its middle; revision 4 drops the anchors (connector midpoints collide, 31 %, and the port graph does not need them)."),
    ("focal BCE for the heading bins and the anchor / split / merge heats;", "focal BCE for the heading bins and the port heats, L1 for the port offsets, cosine for the port headings;"),
    ("The anchor and window-edge port peaks (an index choice; the values carry the gradient); the candidate port pairs within 60 m (an index choice); the output thresholds (existence 0.5, connector 0.5).",
     "The port peaks (an index choice; the values carry the gradient); the candidate port pairs within 60 m (an index choice); the output threshold (connection 0.5)."),
    ("anchor / split / merge key-point F1 within 3 m", "port F1 within 3 m per type (split, merge, start, end, window edge)"),
    ("the GT anchors through the decoder's target code reproduce the GT graph exactly", "the GT ports and GT pairs through the curve code reproduce the GT graph"),
    ("(heading bins, lane anchors, ports, B-spline fits, connector pairs, neighbour pairs)", "(heading bins, ports by type after dropping the pass-through ports, Bezier fits with inner points, positive pairs, neighbour pairs)"),
    ("the representation oracle (GT anchors + GT ports + GT pairs through the curve code) reproduces the GT graph: B-spline fit of the lane pieces within 0.3 m p90 of the GT points, Bezier connectors with the fitted handles within 0.3 m p90 (measured already: 0.26 m), junction F1 > 0.95 (layer 2)",
     "the representation oracle (GT ports + GT pairs through the curve code) reproduces the GT graph: every piece within 0.5 m (measured already for single pieces: 99.5 %; merged chains need the inner points: 86 % with none), junction F1 > 0.95 (layer 2)"),
    ("2. Matching-free anchors first; bipartite matching in the loss only as the fallback", "2. Matching-free (ports as key points, pairs by location); bipartite matching in the loss only as the fallback"),
]
for o, n in rep:
    assert o in s, o[:70]
    s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
