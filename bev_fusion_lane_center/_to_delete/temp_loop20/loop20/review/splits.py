import json
d=json.load(open('data/splits.json'))
print(list(d)[:10])
for k,v in d.items():
    try: print(k,len(v))
    except: print(k,v)
