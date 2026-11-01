import csv,sys
for r in csv.DictReader(open(sys.argv[1],encoding='utf-8')):
    if any(r['id'].startswith(p) for p in sys.argv[2].split(',')):
        print('##',r['id'],r['status'],r['started'][:16],'seed',r['seed'])
        print(' H:',r['hypothesis'][:300])
        print(' P:',r['prediction'][:250],'| RULE:',r['rule'][:250])
        print(' R:',r['result'][:600])
        print(' D:',r['decision'][:400])
        if r['notes']: print(' N:',r['notes'][:400])
