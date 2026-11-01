import json
g=json.load(open('runs/eval/e25_j1phd2_s0/m143_grid.json'))
print(g['name'],g['tau'],g['vtx_r'])
gr=g['grid']
print(type(gr), list(gr)[:3] if isinstance(gr,dict) else gr[:2])
