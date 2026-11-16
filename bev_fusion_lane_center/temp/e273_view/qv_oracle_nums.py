import qgraph_view as QV
for sid, i in ((11, 12), (39, 20), (23, 25)):
    name, toks = QV.val_scenes()[sid]; tok = toks[i]; g = QV.load_gt(tok)
    sn = QV.scorer_numbers(QV.make_pred(g, "oracle", tok), g, tok)
    print(name, i, {k: round(sn[k], 3) for k in ("node_F1", "F1@1.0", "Q", "lanes_found", "topo_F1", "topo_P", "topo_R", "junction_F1", "lane_acc", "road_acc", "dup_frac", "pieces_per_gt_piece") if k in sn})
