p='temp/e273/shorten_report.py'
s=open(p,encoding='utf-8').read()
s=s.replace('''        if m.group(0) == "</div>":
            stack.pop()
        else:''','''        if m.group(0) == "</div>":
            if stack:
                stack.pop()
        else:''')
open(p,'w',encoding='utf-8').write(s)
