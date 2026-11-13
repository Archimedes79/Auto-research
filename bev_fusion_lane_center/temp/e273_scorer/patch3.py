p = "evaluation/qgraph_eval.py"
s = open(p, encoding="utf-8").read()
a = s.index('"""The local scorer')
b = s.index('"""\nimport json')
doc = '''"""The local scorer of the lane head (E273, card section 4): what the head delivers per frame -> the delivered lane graph -> the numbers. Local to the card: `evaluate.py` and the global
protocol are untouched; the graph part is scored by the SAME functions as `evaluate.py` and the lane suite (`lane_suite.py graphs:<name>`), so the reference graph d and the new graph p are
read by one ruler (tested: the stored graphs of recall_sp0_dp give the counts of the stored suite run exactly).

    pred = qformat.oracle_prediction(lane_gt_graph)         # or the head's output: the dict of training/grid/qformat.py (nodes, links, assign)
    graph, bk = readout(pred)                                # the output step: e >= 0.5, p >= 0.5, one piece per section -> LaneGraph + bookkeeping
    frames = val_frames()                                    # the 873 VAL frames (lane suite order): frame.gt = old graph GT, frame.lane() = lane GT of autogt
    graph_metrics(graph, frames[0])                          # HEADLINE: Q, F1@1.0, lanes_found, topo P / R / F1, junction P / R / F1, dup_frac, R@1_far, vert100, pieces_per_gt_piece ...
    node_metrics(pred, frames[0].lane()); assign_metrics(pred, frames[0].lane())
    out = score_frames(preds, frames)                        # {'graph', 'nodes', 'assign'} pooled over frames the project's way (micro counts); records=True: + the per-frame counts for `paired`

Readout (the named OUTPUT step, no loss goes through it). Kept nodes: e >= thr_e; kept links: p >= thr_link between kept nodes (one link per node pair, the most probable). A SECTION (the
delivered lane piece) is a set of nodes with one `section` value; its polyline is the chain of Hermite curves (qtargets.hermite, handles a, b; `curves` if the prediction carries them) of the kept
links INSIDE the section, sampled every 0.5 m (a section whose kept links branch or break gives one polyline per chain, all with that section id). A kept link between two sections is a connector
piece of its own (its Hermite curve: the connectors through the intersections carry no node, so their geometry is exactly these links); edges of the graph join a piece to every piece that
starts at its end node. `bk` = the bookkeeping (also graph.meta): kept nodes, kept links, `cross` (link joins two sections), per piece kind / section / links / nodes, the kept nodes with sections.

Numbers (from ADDITIVE counts per frame, pooled by summing, so any set of frames can be scored again; `*_counts` -> `pool_counts` -> `*_numbers`):
  graph_metrics   `sparse_lanes/bootstrap.py` (Q, S, J, F1@t, lanes_found, split6 / merge6; vert100 = unique kept node ids per 100 m of delivered polyline, vert100_dp = Douglas-Peucker vertices
                  of the same polylines), `topo_metrics.py` (topo P / R / F1, dup_frac, R@1 by range), `lane_task/measures.py` topo_counts (junction P / R / F1 within 3 m: the lane suite's topo
                  metric), all against the OLD graph GT of the frame (`graph_gt_val.pkl`: the GT the lane suite scores d with; the lane GT agrees with it within 0.3 m); plus pieces_per_gt_piece
                  (distinct sections among the nodes on one GT piece, mean over GT pieces with a node) and id_switches_per_100m (changes of the section along a GT piece, per 100 m of GT lane).
                  A node belongs to the GT piece of its nearest GT lane point when that is within 2 m and no other piece has a point within TIE (0.3 m) of it (shared junction points are left out).
  node_metrics    F1 of node existence within 2 m (greedy one-to-one matching, nearest pairs first: a metric only), median position / direction error of the matched nodes, event P / R per flag
                  within 3 m, short / long link P / R and out-degree accuracy on matched nodes, nodes per 100 m of GT lane.
  assign_metrics  lane_acc (e-weighted share of the nodes within 1 m of a GT lane whose most probable map piece is a piece of that lane; a GT piece the map lacks means the dustbin), dust P / R
                  (nodes farther than 2 m from every GT lane, or on a piece the map lacks, belong to the dustbin), purity (mean over GT lanes of the share on the majority map piece: consistency,
                  not correctness), road_acc (the same at road level). A node at a point shared by several GT pieces accepts any of them.
A ratio with an empty denominator is 0.0 when the other side is not empty (an empty prediction scores 0, not 1), NaN when both are empty. Q of an empty graph is 0.15 / 0.85 (its false-length term).
Self-test: python qgraph_eval.py (a toy frame); real frames: evaluation/tests/test_qgraph_eval.py (wsl bash tools/env/_py14.sh test_qgraph_eval.py).
'''
s = s[:a] + doc + s[b:]
open(p, "w", encoding="utf-8").write(s)
