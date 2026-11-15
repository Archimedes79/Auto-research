import csv, sys
rows = list(csv.reader(open('/mnt/c/Auto-research/bev_fusion/records/LEDGER.csv', encoding='utf-8')))
hdr = rows[0]
print(hdr)
for r in rows[1:]:
    if r[0] in ('t23.C_s0', 't23.C_s1'):
        print('-----', r[0])
        for h, v in zip(hdr, r):
            if v:
                print(f'  {h}: {v[:900]}')
