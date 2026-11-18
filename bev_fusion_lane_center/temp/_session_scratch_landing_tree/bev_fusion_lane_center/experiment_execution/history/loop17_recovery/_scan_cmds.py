"""list ALL Bash / PowerShell commands of the transcript that mention a given file and look like they modify it (wide pattern)"""
import json
import re
import sys

src = sys.argv[1]
names = sys.argv[2:]
pat = re.compile(r"write_text|\.write\(|replace\(|re\.sub|sed |perl |node |tee |>>|Set-Content|Add-Content|-replace|Out-File|python[0-9.]* -|<<|truncate|mv |cp |rm |git ")
with open(src, encoding="utf-8", errors="replace") as f:
    for ln, line in enumerate(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        m = d.get("message")
        if not isinstance(m, dict) or m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue
        for b in m["content"]:
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") in ("Bash", "PowerShell"):
                cmd = b["input"].get("command", "")
                hit = [n for n in names if n in cmd]
                if hit and pat.search(cmd):
                    one = re.sub(r"\s+", " ", cmd)
                    print(f"[{ln}] {','.join(hit)} :: {one[:230]}")
