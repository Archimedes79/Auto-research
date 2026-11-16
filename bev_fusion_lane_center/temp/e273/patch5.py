p='temp/e273/shorten_report.py'
s=open(p,encoding='utf-8').read()
s=s.replace('''    h = drop_h3(h, t)
h = h.replace("<h2>Status board''','''    try:
        h = drop_h3(h, t)
    except AssertionError:
        print("already gone:", t)
h = h.replace("<h2>Status board''')
open(p,'w',encoding='utf-8').write(s)
