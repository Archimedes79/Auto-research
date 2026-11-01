import json
g=json.load(open('runs/eval/e25_j1phd2_s0/m143_grid.json'))
import collections
t=collections.defaultdict(list)
for r in g['grid']: t[r['tau']].append((r['vtx_r'],round(r['vert100'],2),round(r['Q'],4),round(r['lanes_found'],3)))
for k,v in sorted(t.items()): print(k,v)
