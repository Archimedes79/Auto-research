import json,glob,os
for p in sorted(glob.glob('runs/eval/e24_j1*/val.json')):
    d=json.load(open(p))
    r=d['registry']; 
    print(os.path.basename(os.path.dirname(p)),d['frames'],d['split'],d.get('subset'),r.get('tau'),r['asm'].get('fork_prune'),r['asm'].get('vtx_r'),r['set'],d['info'].get('tau'), r['run'], d['commit'], d['date'], round(d['metrics']['vert100'],2))
