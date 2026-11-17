import pickle
d=pickle.load(open("../../runs/e365_s1_s0/token_val.pkl","rb"))
print(type(d), list(d.keys())[:12] if isinstance(d,dict) else len(d))
fr=d["frames"] if isinstance(d,dict) and "frames" in d else d
f=fr[0]
print({k:(getattr(v,'shape',None) or type(v).__name__) for k,v in f.items()})
