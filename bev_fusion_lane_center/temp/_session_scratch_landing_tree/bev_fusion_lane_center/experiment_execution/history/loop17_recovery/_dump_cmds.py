"""dump the full Bash / PowerShell commands at given transcript lines (review before replaying them in a sandbox)"""
import json
import sys

src, out = sys.argv[1], sys.argv[2]
want = set(int(x) for x in sys.argv[3:])
with open(src, encoding="utf-8", errors="replace") as f, open(out, "w", encoding="utf-8") as o:
    for ln, line in enumerate(f):
        if ln not in want:
            continue
        d = json.loads(line)
        for b in d["message"]["content"]:
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") in ("Bash", "PowerShell"):
                o.write(f"=== line {ln} id {b['id']}\n{b['input'].get('command', '')}\n\n")
print("done")
