import json,sys
for n in ['e25_j1phd2_s0','e24_j1ex_s0','e24_j1ex_s1']:
    d=json.load(open(f'runs/eval/{n}/val_m143.json'))
    print(n,{k:(v if not isinstance(v,(dict,list)) else '...') for k,v in d.items()})
    for k,v in d.items():
        if isinstance(v,dict) and k not in ('metrics',): print(' ',k,{a:b for a,b in v.items() if not isinstance(b,(dict,list))} )
    m=d['metrics']; print(' ',{k:round(m[k],4) for k in ['Q','vert100','lanes_found','R@1_far','topo_F1','F1@1.0']})
