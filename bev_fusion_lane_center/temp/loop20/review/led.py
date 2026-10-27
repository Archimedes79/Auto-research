import csv
r=list(csv.DictReader(open('LEDGER.csv',encoding='utf8')))
print(r[0].keys())
for x in r:
    if x['id'].startswith(('E255','E256','E253')):
        print('-----',x['id'],x.get('status'))
        for k,v in x.items():
            if v and k in ('result','decision','note','notes','prediction','rule'): print(k,':',v[:1500])
