"""One-off: replace section 6 of plan.md by the text of a file (everything from '## 6. ' up to '## 7. ').
Run from bev_fusion: wsl python3 temp/setup/replace_section6.py <new section file>
"""
import sys

new = open(sys.argv[1], encoding="utf-8").read()
text = open("plan.md", encoding="utf-8").read()
a, b = text.index("\n## 6. "), text.index("\n## 7. ")
open("plan.md", "w", encoding="utf-8", newline="\n").write(text[:a + 1] + new.rstrip("\n") + "\n" + text[b:])
print("section 6 replaced")
