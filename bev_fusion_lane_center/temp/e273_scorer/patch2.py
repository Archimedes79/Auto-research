p = "evaluation/qgraph_eval.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    c = dict(rec=SB.frame_records([raw], [gt])[0], topo=TM.frame_record(raw, gt),''', '''    c = dict(rec=SB.frame_records([raw], [gt])[0], topo=TM.frame_record(raw, gt), dp=np.array(SB._vstats(raw, False)),''')
s = s.replace('''    """the additive counts of one frame: rec (bootstrap.frame_records: Q, S, J, F1, lanes found ... as `lane_suite.LcrMetric`), topo (topo_metrics.frame_record: as `LcrTopoMetric`),
    lane (lane_task topo_counts: junctions, as `LaneTopoMetric`), sec (section counts; only for a readout graph and a frame with a stored lane GT)"""''', '''    """the additive counts of one frame: rec (bootstrap.frame_records: Q, S, J, F1, lanes found ... as `lane_suite.LcrMetric`), topo (topo_metrics.frame_record: as `LcrTopoMetric`),
    lane (lane_task topo_counts: junctions, as `LaneTopoMetric`), dp (Douglas-Peucker vertices and length of the polylines, for vert100_dp), sec (section counts; only for a readout graph
    and a frame with a stored lane GT)"""''')
s = s.replace('''    out.update({k: float(v) for k, v in MS.topo_numbers(lane).items() if k.startswith("junction")})
''', '''    out.update({k: float(v) for k, v in MS.topo_numbers(lane).items() if k.startswith("junction")})
    dp = np.sum([c["dp"] for c in cs], 0)
    out["vert100_dp"] = float(100.0 * dp[0] / max(dp[1], 1e-9))              # the vertices the delivered polylines need at 0.3 m (the project's measure for graphs without node ids)
''')
s = s.replace('''RAGGED = ("pos_err", "dir_err")   # counts that are concatenated, not summed
''', '''RAGGED = ("pos_err", "dir_err")   # counts that are concatenated, not summed
HEADLINE = ("Q", "F1@1.0", "lanes_found", "topo_F1", "topo_P", "topo_R", "junction_F1", "junction_P", "junction_R", "dup_frac", "R@1_far", "vert100", "vert100_dp", "pieces_per_gt_piece",
            "id_switches_per_100m")
''')
# release the caches of a frame after scoring
s = s.replace('''    for p, f in zip(preds, frames):
        if "pts" in p:
            rec["graph"].append(graph_counts(p, f))
            continue
        graph, _ = readout(p, thr_e, thr_link)
        rec["graph"].append(graph_counts(graph, f))
        try:
            ref = f.ref()
        except FileNotFoundError:
            miss += 1
            continue
        rec["node"].append(node_counts(p, ref, thr_e, thr_link))
        a = assign_counts(p, ref, thr_e)
        if a is not None:
            rec["assign"].append(a)
''', '''    for p, f in zip(preds, frames):
        cached = getattr(f, "_ref", None) is not None                        # a frame loaded before keeps its lane GT; one loaded here lets it go (873 of them are large)
        if "pts" in p:
            rec["graph"].append(graph_counts(p, f))
            continue
        graph, _ = readout(p, thr_e, thr_link)
        rec["graph"].append(graph_counts(graph, f))
        try:
            ref = f.ref()
        except FileNotFoundError:
            miss += 1
            continue
        rec["node"].append(node_counts(p, ref, thr_e, thr_link))
        a = assign_counts(p, ref, thr_e)
        if a is not None:
            rec["assign"].append(a)
        if not cached:
            f._lane = f._ref = None
''')
open(p, "w", encoding="utf-8").write(s)
