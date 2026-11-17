import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
a = '''            for k in ms:
                ms[k].append(row[k])
    load1, gpu1'''
assert s.count(a) == 1
s = s.replace(a, '''            for k in ms:
                if row[k]:                                                                # B' alone and the whole path only with --b-run
                    ms[k].append(row[k])
    load1, gpu1''')
open(p, "w", encoding="utf-8").write(s)
