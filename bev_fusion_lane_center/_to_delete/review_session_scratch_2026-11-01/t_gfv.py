import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'evaluation/sparse_lanes'))
import numpy as np
import sparse_metrics as SM
import graph_eval as GE
# chain 0 -> 1 -> 2 (split at 2) -> 3 -> 4 ; 2 -> 5 -> 6
xy = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [30, 5], [40, 10]], float)
E = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)]
g = SM.graph_from_vertices(xy, E)
print('split after 1-in/1-out chain:', g['nodes'], g['edges'])
print('junctions (split pts, merge pts):', GE.junctions(g))
# merge case: 0->1->2, 5->2? merge at 2 then 2->3->4
E2 = [(0, 1), (1, 2), (5, 6), (6, 2), (2, 3), (3, 4)]
g2 = SM.graph_from_vertices(xy, E2)
print('merge:', g2['nodes'], g2['edges'])
# pure chain whose first vertex has in=1 from a vertex of out=1 but a cycle
