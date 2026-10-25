"""One-off: replace section 2 (camera) of plan.md with the text of a given file (everything from '## 2.' up to '## 3.').

Run from the bev_fusion folder: wsl python3 temp/setup/replace_section2.py <new section file>
"""
import sys

plan = "plan.md"
new = open(sys.argv[1], encoding="utf-8").read()
text = open(plan, encoding="utf-8").read()
a = text.index("\n## 2. ")
b = text.index("\n## 3. ")
text = text[:a + 1] + new.rstrip("\n") + "\n" + text[b:]
open(plan, "w", encoding="utf-8", newline="\n").write(text)
print("section 2 replaced:", len(new), "characters")
