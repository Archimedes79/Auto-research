import pickle,sys,json
p=sys.argv[1]
d=pickle.load(open(p,'rb'))
print(type(d), len(d) if hasattr(d,'__len__') else '')
if isinstance(d,dict):
    for k,v in list(d.items())[:10]:
        print(k, type(v), (len(v) if hasattr(v,'__len__') else v))
        if isinstance(v,(list,tuple)) and v: print('  ',type(v[0]), str(v[0])[:300])
        if isinstance(v,dict):
            for kk,vv in list(v.items())[:6]: print('   ',kk,str(vv)[:200])
elif isinstance(d,list):
    print(str(d[0])[:600])
