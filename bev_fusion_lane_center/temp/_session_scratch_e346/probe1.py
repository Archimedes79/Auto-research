import time, numpy as np
import qgraph_eval as QE
t=time.time()
fr = QE.val_frames()
print(len(fr), time.time()-t)
f = fr[0]
print(type(f.gt), list(f.gt.keys()) if isinstance(f.gt, dict) else None)
for k,v in f.gt.items():
    print(k, type(v), (len(v) if hasattr(v,'__len__') else v))
print([ (len(p)) for p in f.gt['pts']][:10], f.gt['edges'][:10])
t=time.time()
g = f.lane()
print('lane load', time.time()-t)
print(list(g.keys()))
p = g['pieces'][0]
print({k:(type(v).__name__, (len(v) if hasattr(v,'__len__') else v)) for k,v in p.items()})
print(p['segments'][:3])
print(g['events'][0])
print(QE.stored_share(fr))
