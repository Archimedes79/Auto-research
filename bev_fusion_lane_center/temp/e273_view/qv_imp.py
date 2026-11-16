import time, sys, resource
t0 = time.time()
import qgraph_eval as QE
print("import", round(time.time() - t0, 2))
t0 = time.time(); fr = QE.val_frames(); print("val_frames", round(time.time() - t0, 2), len(fr), "maxrss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1000)
import qgraph_view as QV
tok = QV.val_scenes()[11][1][12]
g = QV.load_gt(tok)
for kind in ("oracle", "degraded"):
    pred = QV.make_pred(g, kind, tok)
    t0 = time.time()
    graph, bk = QE.readout(pred)
    f = next(x for x in fr if x.token == tok)
    gm = QE.graph_metrics(graph, f)
    nm = QE.node_metrics(pred, g); am = QE.assign_metrics(pred, g)
    print(kind, round(time.time() - t0, 2), "s", {k: round(gm[k], 3) for k in ("F1@1.0", "Q", "lanes_found", "topo_F1", "junction_F1") if k in gm}, "node_F1", round(nm["node_F1"], 3), "lane_acc", round(am["lane_acc"], 3))
