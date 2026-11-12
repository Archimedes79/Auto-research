import re,sys
h=open(sys.argv[1],encoding='utf-8').read()
h=re.sub(r'<svg.*?</svg>','[SVG]',h,flags=re.S)
for key in sys.argv[2:]:
    k=h.index(key); print('=====',key); print(h[k-900:k+300])
