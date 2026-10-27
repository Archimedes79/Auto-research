import json
d=json.load(open('runs/eval/e24_j1rc_s2/val.json'))
def walk(x,p='',depth=0):
    if isinstance(x,dict):
        for k,v in x.items():
            if isinstance(v,(dict,list)) and depth<2: walk(v,p+'/'+k,depth+1)
            else: print(p+'/'+k, str(v)[:80])
walk(d)
