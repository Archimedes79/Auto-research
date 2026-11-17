import json,sys
d=json.load(open("temp/jr/experiments_table.json"))
rows=d if isinstance(d,list) else d.get("rows",d)
want=set(sys.argv[1:])
for e in rows:
    if e.get("id") in want:
        print(e["id"],"|",{k:(str(v)[:420]) for k,v in e.items() if k!="id"})
        print()
