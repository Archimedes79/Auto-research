import qformat, qgraph_eval as QE
fr = QE.val_frames()[::13]
out = QE.score_frames([qformat.oracle_prediction(f.lane()) for f in fr], fr)
g = out["graph"]
print("oracle vert100 (node ids)", round(g["vert100"], 2), "vert100_dp", round(g["vert100_dp"], 2), "nodes per 100 m of GT lane", round(out["nodes"]["nodes_per_100m"], 2))
print({k: round(g[k], 4) for k in QE.HEADLINE})
