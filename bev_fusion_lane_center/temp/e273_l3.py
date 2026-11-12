"""one-off: E273 card, GT status after the lane GT l3 (R1, R3 in) and the undilated intersection polygons"""
p = "experiments/E273_pyramid-head-soft-graph-map-assignment/experiment.md"
s = open(p, encoding="utf-8").read()
rep = [
    ("                                               2 m:   road, intersection (dilated by 3 m); the 2 m feature map is the node grid",
     "                                               2 m:   road, intersection (polygon rasterised at 2 m); the 2 m feature map is the node grid"),
    ("- **2 m** (50 x 50): road (drivable area) and intersection (the polygon dilated by 3 m: the stop points sit just inside it); the road-level graph is a v2 item (section 8).",
     "- **2 m** (50 x 50): road (drivable area) and intersection: the GT polygons rasterised at 2 m, a cell counts when any part of it is covered (that is the slight enlargement); Mirko told the GT chat to leave the dilation out of the GT, so a further dilation is a loader parameter, default 0 m; the road-level graph is a v2 item (section 8)."),
    ("**l2** adds the drivable-area and intersection polygons; the front GT (c643a7539d) is untouched. The full store builds in the background (about 45-60 min, nice 15).",
     "**l3** (tag `autogt/lane-l3`, signature 3576fa1f7e, commit 4df1032, local) adds the drivable-area and intersection polygons (`g[\"areas\"]`, not dilated), the two additions of this plan (`exit` events, `segments` per piece) and review fixes (joined-piece stops now name their continuation, aligned direction rows, `opposite` on both sides, short parts with a natural end are kept); the front GT (c643a7539d) is untouched. The full store (850 sequences) builds with l3 in the background (about 45 min, nice 15). The piece id `piece` is the same in every frame (use it for the assignment targets); `id` differs only for a part that comes back into the window (`<piece>#1`)."),
    ("- **R1 exit events:** the mirror of `stop`: the first point of a lane whose predecessor connector lies inside an intersection (same fields: `on`, `s`, the intersection test of `stop`). Needed for the type \"exit\" of the nodes.\n- **R2 (done in l2):** area polygons.\n- **R3 segments per piece:** `segments = [{token, kind, i0, i1, in_intersection}]` (index ranges of the piece's points per original map piece, with the test result), so that the loader can drop the points of connectors inside intersections and keep the points of connectors outside them (16-20 % of the connectors are plain continuations).",
     "- **R1 exit events (done in l3):** the mirror of `stop`: the start of a lane whose predecessor connector lies inside an intersection (fields `on`, `s`, `n_pred`, `in`, `out`, `dirs_in`, `dirs_out`; `pos` is the joint vertex). Needed for the type \"exit\" of the nodes.\n- **R2 (done in l2 / l3):** area polygons (`areas.road`, `areas.intersection`, polygons with holes, not dilated).\n- **R3 segments per piece (done in l3):** `segments = [{token, kind, i0, i1, in_intersection}]` (contiguous index ranges of the part's points per original map piece, with the test result; False for lanes), so that the loader can drop the points of connectors inside intersections and keep the points of connectors outside them (16-20 % of the connectors are plain continuations)."),
    ("lane GT version and signature (autogt/lane-l1 / l2, 42a3d08f8f / 36847fdc49)", "lane GT version and signature (autogt/lane-l3, 3576fa1f7e)"),
    ("| 0 | GT requests R1, R3 to the GT chat; target builder, map simulator, oracle |", "| 0 | (R1, R3 are in l3, the l3 store is building); target builder, map simulator, oracle |"),
    ("Dependencies on other chats: the GT chat (R1, R3, the lane store);", "Dependencies on other chats: the GT chat (the l3 lane store);"),
]
for o, n in rep:
    assert o in s, o[:70]
    s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
