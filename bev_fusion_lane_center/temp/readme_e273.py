"""one-off: README rows and test registration of the E273 modules (training/grid, training/README, run_all_tests.sh)"""
p = "training/grid/README.md"
s = open(p, encoding="utf-8").read()
rows = """| `qtargets.py` | E273: nodes on the 2 m grid, links by contraction along the pieces, section starts, `grid_targets` (the 50 x 50 maps of the node head), `decode` / `fit_long_scales` (Hermite curves: the representation oracle), from the lane GT of autogt (`autogt.lane_gt`) | new (card E273 section 3.3) | `python qtargets.py`, `test_qtargets.py` (real frames: lane points 99.9 %, connectors 98.7 % within 1 m) |
| `qformat.py` | E273: THE CONTRACT of what the lane head delivers per frame (nodes, links, assignment, maps) and `oracle_prediction(g)` (a perfect network made from the GT) for the scorer and the pictures | new | `python qformat.py` |
| `map_sim.py` | E273: the map side: the lane GT pieces of a frame with simulated errors (translation, yaw, thinned shape points, dropped pieces, wrong lane counts), tokens every 2 m, relations succ / pred / left / right / opposite | idea of `training/data/sd_lib.py` (degrade_sd) at lane level | `python map_sim.py` |
| `qraster.py` | E273: the pixel targets drawn from the lane GT vectors (0.5 m lines with distance, 1 m event heats and centre points, 2 m road and intersection) | new | `python qraster.py`, `test_qraster.py` |
| `pyramid.py` | E273 stage A: E271's body with three taps (0.5 m lines, 1 m centres + event heats, 2 m road / intersection) and the feature maps of the node grid; `load_e271` | `head.py` | `python pyramid.py` |
| `qnodes.py` | E273 stage B: one node per 2 m cell (existence, flags, offset, direction, section start, long-link scores), neighbourhood attention, refinement by sampling the fine features | new | `python qnodes.py` |
| `qlinks.py` | E273 stage C: the connection matrix (short links in the 8 m neighbourhood, long links between the top-K entries and exits), Hermite curves with learned handle scales, evidence along the curve | new | `python qlinks.py` |
| `sections.py` | E273 stage D: sections (ids) as a soft absorbing walk backwards on the link matrix (a linear solve) and its loss | new | `python sections.py` |
| `qassign.py` | E273 stage E: map encoder (graph transformer over the pieces along succ / pred / left / right / opposite) and the assignment (cross-attention + E272's geometric term + dustbin + soft registration) | `assign.py`, SMERF / SEPT / MAT | `python qassign.py` |
| `qgraph.py` | E273: the whole lane head as one module, the loss of every phase, trainable groups | the parts above | `python qgraph.py` (every phase: gradient reaches the trainable groups, none the frozen ones) |
"""
anchor = "| `test_oracle.py` |"
assert anchor in s
s = s.replace(anchor, rows + anchor, 1)
s = s.replace("Trainers: `../train/grid_train.py` (E271, the head alone) and `../train/chain_train.py` (E272, the chain: blocks first with the head frozen, then joint), both on `../trainer.py`,",
              "Trainers: `../train/grid_train.py` (E271, the head alone), `../train/chain_train.py` (E272, the chain: blocks first with the head frozen, then joint) and `../train/qgraph_train.py` (E273: the lane head phase by phase with a gate after each, `../train/qgates.py`), all on `../trainer.py`; data of E273: `../data/lane_gt_data.py`;")
open(p, "w", encoding="utf-8").write(s)

p = "training/README.md"
s = open(p, encoding="utf-8").read()
s = s.replace("`chain_train.py` (E272: the whole chain, blocks then joint) |", "`chain_train.py` (E272: the whole chain, blocks then joint), `qgraph_train.py` + `qgates.py` (E273: the lane head, pixel maps -> nodes -> links -> sections -> map assignment, a gate after every phase) |")
open(p, "w", encoding="utf-8").write(s)

p = "tools/maintenance/run_all_tests.sh"
s = open(p, encoding="utf-8").read()
add = """run "E273 lane head: targets, nodes, links, Hermite oracle (synthetic)" grid/qtargets.py
run "E273 lane head: prediction contract" grid/qformat.py
run "E273 lane head: map simulator" grid/map_sim.py
run "E273 lane head: pyramid head" grid/pyramid.py
run "E273 lane head: node head" grid/qnodes.py
run "E273 lane head: link head" grid/qlinks.py
run "E273 lane head: sections" grid/sections.py
run "E273 lane head: map encoder + assignment" grid/qassign.py
run "E273 lane head: all phases, gradients" grid/qgraph.py
"""
anchor = 'run "grid oracle (G0): the GT through the chain reproduces itself" grid/test_oracle.py\n'
assert anchor in s
s = s.replace(anchor, anchor + add, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
