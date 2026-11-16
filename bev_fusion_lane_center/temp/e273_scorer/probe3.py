import json, os, sys, numpy as np
import bflc_paths as BP
sys.path.insert(0, os.path.join(BP.REPO, "autoresearch"))
from evalsuite.store import RunStore
import lane_suite as LS
from lane_task import LaneTopoMetric, LaneSparseMetric
RUN = f"{BP.ROOT}/runs/suite/graphs_recall_sp0_dp/quant/graphs_recall_sp0_dp_carry_20261104_181557"
st = RunStore.open(RUN)
print(len(st.frames), st.frames[0]["key"], st.manifest["metrics"].keys())
cfg = LS.config()
for name, m in (("lcr", LS.LcrMetric()), ("lcr_topo", LS.LcrTopoMetric()), ("geo", LaneSparseMetric(cfg)), ("topo", LaneTopoMetric(cfg))):
    c = st.counts(name)
    print(name, {k: np.shape(v) for k, v in c[0].items()})
    r = m.reduce(c)
    print(name, {k: round(v, 4) for k, v in r.items() if k in ("Q","S","F1@1.0","J","lanes_found","vert100","split6_P","split6_R","split6_F1","merge6_F1","topo_F1","topo_P","topo_R","dup_frac","R@1_far","junction_P","junction_R","junction_F1","chamfer","F1_1to1","len_ratio")})
