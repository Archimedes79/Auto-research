p = "temp/e276/e276_build_report.py"
s = open(p, encoding="utf-8").read()
a = s.index('css = """')
b = s.index('"""', a + 10) + 3
css = '''css = """
:root{--bg:#fbfbf9;--fg:#1d2329;--mut:#5b6670;--line:#d8dde2;--acc:#1f6feb;--ok:#2a7d3f;--bad:#b3261e;--pend:#8a5f00}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14181c;--fg:#e6e9ec;--mut:#9aa5af;--line:#2b333b;--acc:#58a6ff;--ok:#56d364;--bad:#ff7b72;--pend:#e3b341;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#14181c;--fg:#e6e9ec;--mut:#9aa5af;--line:#2b333b;--acc:#58a6ff;--ok:#56d364;--bad:#ff7b72;--pend:#e3b341;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;padding-block:0 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
main{max-width:980px;margin:0 auto}h1{font-size:1.6rem;margin:28px 0 4px;text-wrap:balance}h2{font-size:1.2rem;margin:32px 0 6px;border-top:1px solid var(--line);padding-top:16px;text-wrap:balance}h3{font-size:1.02rem;margin:20px 0 4px}
p,li{max-width:80ch}.mut{color:var(--mut)}.pending{color:var(--pend)}table{border-collapse:collapse;margin:10px 0;display:block;max-width:100%;overflow-x:auto;font-variant-numeric:tabular-nums}th,td{border:1px solid var(--line);padding:4px 10px;text-align:left}
figure{margin:12px 0}img{max-width:100%;height:auto}figcaption{color:var(--mut);font-size:.9rem}.note{border-left:3px solid var(--acc);padding:2px 12px;margin:10px 0}.box{border:1px solid var(--line);border-radius:6px;padding:4px 16px;margin:14px 0}
pre{white-space:pre-wrap;font-size:.8rem;overflow-x:auto}
"""'''
s = s[:a] + css + s[b:]
# fragment form: title + meta + style, then the page content (the Artifact tool adds the skeleton)
old = s[s.index('parts = [f"<!doctype html>'):s.index('"<h1>Spurmitte: what the night showed</h1>"')]
new = '''parts = [f"<title>Spurmitte E276 report</title><meta name='claude:viewport-fit-sides' content='cover'><style>{css}</style><main>",
         '''
s = s.replace(old, new)
s = s.replace('</ul></main></body></html>")', '</ul></main>")')
open(p, "w", encoding="utf-8").write(s)
print("ok")
