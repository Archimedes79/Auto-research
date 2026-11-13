import os, sys
sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation")); sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "training/grid"))
import numpy as np, qformat, qgraph_eval as QE
g, old = QE.toy_lane_gt()
p = qformat.oracle_prediction(g)
print(np.round(p["nodes"]["pos"], 2).tolist()); print(p["nodes"]["section"].tolist()); print(list(zip(p["links"]["src"].tolist(), p["links"]["dst"].tolist())))
