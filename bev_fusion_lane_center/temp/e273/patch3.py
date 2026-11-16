p='temp/e273/shorten_report.py'
s=open(p,encoding='utf-8').read()
s=s.replace('def drop_h3(h, title_start, section=None):','def bal(h):\n    return h.count("<div"), h.count("</div>")\n\n\ndef drop_h3(h, title_start, section=None):\n    print(title_start[:30], bal(h), end=" -> ")\n    r = _drop_h3(h, title_start)\n    print(bal(r))\n    return r\n\n\ndef _drop_h3(h, title_start):')
s=s.replace('h = drop_section(h, "gt")','print("start", bal(h))\nh = drop_section(h, "gt")\nprint("gt", bal(h))')
open(p,'w',encoding='utf-8').write(s)
