import json
for n in ['e25_j1phd2_s0','e24_j1ex_s0','e24_j1ex_s1','e24_j1eq_s0','e24_j1eq_s1']:
    d=json.load(open(f'runs/eval/{n}/val.json')); m=d['metrics']
    print(n,d['frames'],d['split'],d['registry']['tau'],d['registry']['asm'].get('vtx_r') if 'asm' in d['registry'] else None,d['info'].get('tau'),{k:round(m[k],4) for k in ['Q','vert100','lanes_found','R@1_far','topo_F1','F1@1.0']}, d['commit'],d['date'])
    g=json.load(open(f'runs/eval/{n}/m143_grid.json')) if n.startswith('e25') or 'ex' in n else None
    if g: print('  grid keys',list(g)[:10] if isinstance(g,dict) else type(g))
