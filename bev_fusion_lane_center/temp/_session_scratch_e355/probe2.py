"""throw-away: numbers of one arm on every 97th VAL frame (9 frames), and the time of the parts"""
import sys
import time

import e346_lib as L
import qgraph_eval as QE

arm = sys.argv[1]
fr = QE.val_frames()[::97]
t = time.time()
with L.cell_size(L.GRID[arm]):
    m = L.numbers(L.pool([L.frame_record(arm, f) for f in fr]))
print(arm, f"{time.time() - t:.1f} s for {len(fr)} frames")
for k in ("topo_F1", "lanes_found", "F1@1.0", "junction_F1", "nodes_per_frame", "link_all_d_med", "link_all_d_p90", "wbl_connector_share", "junc_nodes_per_connector", "conn_cov_03", "nn_share_lt20", "nn_share_lt25", "merge_node_share"):
    print(f"  {k} {m[k]:.4f}")
