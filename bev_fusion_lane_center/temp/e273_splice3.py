"""one-off: revision 4, third pass: phase names (piece decoder -> port graph)"""
p = "experiments/E273_pyramid-head-soft-graph-map-assignment/experiment.md"
s = open(p, encoding="utf-8").read()
rep = [
    ("phase 2 piece decoder (head frozen), phase 3 map encoder + assignment (head and decoder frozen)", "phase 2 port graph (head frozen), phase 3 map encoder + assignment (head and port graph frozen)"),
    ("after phase 2 F1@1.0 of the pieces more than 0.05 below d", "after phase 2 F1@1.0 of the delivered pieces more than 0.05 below d"),
    ("| 2 piece decoder | decoder, relation heads | head | points L1 + direction, existence, start / end type, successor and neighbour BCE | 3000 |",
     "| 2 port graph | connection scorer, handle / inner-point MLP, neighbour head | head (its port heats are trained in phase 1) | connection BCE + soft degree loss, handle and inner-point L1, curvature penalty, neighbour BCE | 3000 |"),
    ("| 3 piece decoder | phase 2 | `training/grid/pieces.py` |", "| 3 port graph | phase 2 | `training/grid/port_graph.py` |"),
    ("| CondLaneNet, CenterNet | one anchor (start point / centre) per instance from a heat map, shape decoded from the anchor, no matching | matching-free training of the piece queries |",
     "| CenterNet, SAM-Road, MaGRoad | key points as heat maps without matching; a transformer scoring links between key points; an edge score from statistics along the candidate path | ports as key points, the connection scorer |"),
]
for o, n in rep:
    assert o in s, o[:70]
    s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
