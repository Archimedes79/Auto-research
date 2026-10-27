import json,glob,os
for n in ['champion','recall','fast_recall','control','fast','rc1','e219_rc1','champion_noprune']:
    for f in ['val.json','val_v143.json']:
        p=f'runs/eval/{n}/{f}'
        if not os.path.exists(p): continue
        d=json.load(open(p)); m=d['metrics']
        print(n,f,d['split'],d['frames'],d['registry'].get('run'),d.get('overrides'),{k:round(m.get(k,float('nan')),4) for k in ['Q','S','vert100','topo_F1','lanes_found','R@1_far','J','F1@1.0']},d['info'].get('asm_ms'),d['date'])
