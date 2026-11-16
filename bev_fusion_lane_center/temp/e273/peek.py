import re
h=open('index.html',encoding='utf-8').read()
i=h.index('id="look"'); j=h.index('id="weak"')
s=re.sub(r'<svg.*?</svg>','[SVG]',h[i:j],flags=re.S)
print(s[:3500])
print('....')
k=s.index('scene-0096')
print(s[k-300:k+1800])
n=h.index('<nav'); print(h[n:n+900])
