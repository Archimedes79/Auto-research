import json,glob,os
R=os.environ["BFLC_ROOT"]
c=json.load(open(f"{R}/runs/e324_s_s0/ends_card_snap_thr0.1.json"))
print({k:(v if not isinstance(v,(dict,list)) else type(v).__name__) for k,v in c.items()})
n=c['numbers']; print(list(n.keys()))
for k,v in n.items():
    print(k, {kk:vv for kk,vv in v.items() if kk in ('topo_F1','topo_P','topo_R','lanes_found','junction_F1','dup_frac')} if isinstance(v,dict) else v)
print(c['difference'] if len(str(c['difference']))<1500 else str(c['difference'])[:1500])
for run in ("e324_s_s0","e312_p_s0"):
  for nm in ("snap","attach","snap_nogeom","snap_nogeom_attach"):
    f=f"{R}/runs/{run}/ends_card_{nm}_thr0.1.json"
    if os.path.exists(f):
        c=json.load(open(f)); n=c['numbers']
        print(run,nm,{k:round(v['topo_F1'],4) for k,v in n.items() if isinstance(v,dict) and 'topo_F1' in v}, c.get('seed'), c.get('frames'))
