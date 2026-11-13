p = "experiments/E273_pyramid-head-soft-graph-map-assignment/experiment.md"
s = open(p, encoding="utf-8").read()
rep = [
 ("- **2 m** (50 x 50): road (drivable area) and intersection: the GT polygons rasterised at 2 m, a cell counts when any part of it is covered (that is the slight enlargement); Mirko told the GT chat to leave the dilation out of the GT, so a further dilation is a loader parameter, default 0 m; the road-level graph is a v2 item (section 8).",
  "- **2 m** (50 x 50): road (drivable area) and intersection: the GT polygons rasterised at 2 m, a cell counts when any part of it is covered; Mirko told the GT chat to leave the dilation out of the GT, so the enlargement (\"slightly enlarged\") is a loader parameter `dilate`, **2 m** (measured on 120 frames: with the undilated polygon only 41 % of the stop / exit events lie in an intersection cell (83 % within 4 m) because the stop / exit points are the connector ends, 1-3 m before the polygon edge; 2 m gives 75 % / 91 %, 3 m 83 % / 94 %); the road-level graph is a v2 item (section 8)."),
 ("1. One node slot per 2 m cell (measured).", "0. Mirko's answers of 2026-11-04 night: nodes inside intersections only where there is some form of marker (measured: 0.0 % of the connector points inside intersections lie within 1 m of a painted lane / road divider, 11.9 % within 2 m: practically no painted marker inside, so the marker rule is the event markers stop / exit / split / merge / start / end / window; `qtargets.node_table(marker_dist=...)` keeps painted-divider points as nodes if wanted, default 0); the road level derived from the lanes is accepted (the GT chat changes the GT where it needs to); temporal fusion stays out and becomes a follow-up experiment: cross-attention from the current nodes to the nodes (queries) of earlier frames; the GT defaults stay (boundary = drivable-area outline, corner 30 degrees over 2 m, a stop point with several successors only for the intersection branches).\n1. One node slot per 2 m cell (measured)."),
]
for o, n in rep:
    assert o in s, o[:60]
    s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
