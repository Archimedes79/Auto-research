import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'evaluation/sparse_lanes'))
import numpy as np
import sparse_metrics as SM
import topo_metrics as TM
# trunk 0..20 m (vertices every 2 m), split at x=20: branch A straight, branch B diverging 0.08 m per m
tr = [(x, 0.0) for x in np.arange(0, 21, 2.0)]
A = [(20 + s, 0.0) for s in np.arange(2, 31, 2.0)]
B = [(20 + s, 0.08 * s) for s in np.arange(2, 31, 2.0)]
xy = np.array(tr + A + B, float)
nt, na = len(tr), len(A)
E = [(i, i + 1) for i in range(nt - 1)]
E += [(nt - 1, nt)] + [(nt + i, nt + i + 1) for i in range(na - 1)]
E += [(nt - 1, nt + na)] + [(nt + na + i, nt + na + i + 1) for i in range(na - 1)]
g = SM.graph_from_vertices(xy, E)
print('pieces', len(g['pts']), 'edges', g['edges'])
print('dup (graph_from_vertices):', TM.dup_counts(g))
# the same geometry as three proper pieces
g2 = dict(pts=[xy[:nt], np.vstack([xy[nt - 1], xy[nt:nt + na]]), np.vstack([xy[nt - 1], xy[nt + na:]])], edges=[(0, 1), (0, 2)])
print('dup (three proper pieces):', TM.dup_counts(g2))
