import pickle,sys,json
import numpy as np
p=sys.argv[1]
d=pickle.load(open(p,'rb'))
def show(o,ind=0,maxd=6):
    pad='  '*ind
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,(dict,)):
                print(pad+str(k)+':'); 
                if ind<maxd: show(v,ind+1)
            elif isinstance(v,np.ndarray): print(pad+str(k),'ndarray',v.shape,v.dtype, v.ravel()[:6])
            else: print(pad+str(k),type(v).__name__,str(v)[:100])
show(d['base'][0])
print(d['scenes'][:20])
print(len(set(d['scenes'])))
